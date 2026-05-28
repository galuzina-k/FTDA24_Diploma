import json
import re

from recsys.data.catalog import get_movie_by_id
from recsys.data.inspired import Turn
from recsys.data.title_index import TitleIndex
from recsys.llm.client import LLMClient
from recsys.recommenders.base import BaseRecommender, RecommendationResult
from recsys.retrieval.searcher import Searcher

FAISS_POOL = 20
LLM_RECALL = 20

PREF_EXTRACTION_PROMPT = """\
You analyze a movie recommendation conversation and extract the seeker's preferences.
Output a single JSON object with these fields:
- "genres": list of preferred genres (strings)
- "moods": list of mood/tone keywords (e.g. "dark", "lighthearted", "tense")
- "likes": list of concrete things the seeker wants (themes, actors, eras, styles)
- "dislikes": list of things to avoid
- "search_query": a single dense sentence (max ~30 words) optimized for semantic search over movie plots and metadata

Output ONLY the JSON object. No preamble, no explanation."""

OW_RECALL_PROMPT = """\
You are a movie recommendation assistant. The user describes what they are looking for.
Recommend exactly 20 movies that best match their request, ranked from best to worst fit.
Reply with a JSON array of objects, each with "title" and "year" fields.
Example: [{"title": "The Matrix", "year": "1999"}, ...]
Output ONLY the JSON array. No preamble, no explanation."""

RERANK_PROMPT = """\
You are reranking movie candidates for a seeker based on their preferences.
You are given a numbered list of candidates from two sources: semantic retrieval (faiss) and LLM recall (llm).
Both sources are equally valid — pick the 10 best matches regardless of source, ranked best to worst.
Output ONLY a raw JSON object starting with { and ending with }.
{"ranking": [0, 3, 7, ...]} — a list of exactly 10 integer indices. No explanation."""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    return text


def _parse_json_loose(text: str) -> dict | list | None:
    text = _strip_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            return None
    return None


class HybridRecommender(BaseRecommender):
    name = "hybrid"

    def __init__(
        self,
        llm: LLMClient | None = None,
        searcher: Searcher | None = None,
        title_index: TitleIndex | None = None,
        faiss_pool: int = FAISS_POOL,
        llm_recall: int = LLM_RECALL,
    ):
        self._llm = llm or LLMClient()
        self._searcher = searcher
        self._index = title_index or TitleIndex()
        self._faiss_pool = faiss_pool
        self._llm_recall = llm_recall
        slug = self._llm.model.replace("/", "_").replace("-", "_").replace(".", "_")
        self.name = f"hybrid__{slug}"

    def _get_searcher(self) -> Searcher:
        if self._searcher is None:
            self._searcher = Searcher()
        return self._searcher

    def recommend(
        self,
        dialog_history: list[Turn],
        query: str,
        history_imdb_ids: list[str],
        top_k: int = 10,
    ) -> RecommendationResult:
        seeker_block = self._format_seeker_block(dialog_history)
        history_block = self._format_history_block(history_imdb_ids)

        # Stage 1: extract structured preferences (LLM call 1)
        prefs, prefs_raw = self._extract_preferences(query, seeker_block, history_block)

        # Stage 2a: FAISS retrieval using enriched query
        search_text = self._build_search_text(query, seeker_block, prefs)
        faiss_candidates = self._retrieve_faiss(search_text, history_imdb_ids)

        # Stage 2b: LLM open-world recall (LLM call 2)
        llm_candidates, ow_raw = self._recall_open_world(
            query, seeker_block, history_imdb_ids
        )

        # Stage 3: merge pools, dedup by imdb_id
        candidates = self._merge_candidates(
            faiss_candidates, llm_candidates, history_imdb_ids
        )

        if not candidates:
            return RecommendationResult(
                movie_ids=[],
                explanation="No candidates after merging retrieval and recall.",
                extra={
                    "preferences": prefs,
                    "preferences_raw": prefs_raw,
                    "ow_raw": ow_raw,
                },
            )

        # Stage 4: LLM rerank the merged pool (LLM call 3)
        ranking, rerank_raw = self._rerank(
            prefs, candidates, top_k, seeker_block, query
        )
        picks = self._materialize_picks(ranking, candidates, top_k)

        n_faiss = sum(1 for c in candidates if c["source"] == "faiss")
        n_llm = sum(1 for c in candidates if c["source"] == "llm")
        return RecommendationResult(
            movie_ids=picks,
            explanation=(
                f"Hybrid: prefs → FAISS ({n_faiss}) + LLM recall ({n_llm}) "
                f"= {len(candidates)} candidates → rerank → top {top_k}."
            ),
            extra={
                "preferences": prefs,
                "preferences_raw": prefs_raw,
                "ow_raw": ow_raw,
                "candidates": [
                    {"id": c.get("imdb_id") or c["id"], "source": c["source"]}
                    for c in candidates
                ],
                "rerank_raw": rerank_raw,
                "ranking": ranking,
            },
        )

    # ------------------------------------------------------------------ #
    # Stage 1: preference extraction
    # ------------------------------------------------------------------ #

    def _extract_preferences(
        self, query: str, seeker_block: str, history_block: str
    ) -> tuple[dict, str]:
        sections = []
        if query:
            sections.append(f"Annotated user query:\n{query}")
        if seeker_block:
            sections.append(f"Recent seeker turns:\n{seeker_block}")
        if history_block:
            sections.append(f"Movies mentioned in the dialog:\n{history_block}")
        user_msg = "\n\n".join(sections) if sections else "(no context available)"

        try:
            raw = self._llm.complete(
                [
                    {"role": "system", "content": PREF_EXTRACTION_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
        except Exception as e:
            return {"search_query": query}, f"<error: {e}>"

        parsed = _parse_json_loose(raw)
        if not isinstance(parsed, dict):
            parsed = {"search_query": query}
        for key in ("genres", "moods", "likes", "dislikes"):
            val = parsed.get(key)
            parsed[key] = [str(x) for x in val if x] if isinstance(val, list) else []
        sq = parsed.get("search_query")
        parsed["search_query"] = (
            sq.strip() if isinstance(sq, str) and sq.strip() else query
        )
        return parsed, raw

    # ------------------------------------------------------------------ #
    # Stage 2a: FAISS retrieval
    # ------------------------------------------------------------------ #

    def _build_search_text(self, query: str, seeker_block: str, prefs: dict) -> str:
        base = f"{seeker_block} {query}".strip() if seeker_block else query
        extras: list[str] = []
        for key in ("genres", "moods", "likes"):
            extras.extend(str(v) for v in (prefs.get(key) or []) if v)
        sq = prefs.get("search_query")
        if sq:
            extras.append(sq)
        suffix = " ".join(extras)
        return f"{base} {suffix}".strip() if suffix else base

    def _retrieve_faiss(
        self, search_text: str, history_imdb_ids: list[str]
    ) -> list[dict]:
        seen = set(history_imdb_ids)
        results = self._get_searcher().search(search_text, top_k=self._faiss_pool * 2)
        candidates = []
        for r in results:
            mid = r.get("imdb_id") or r["id"]
            if mid in seen:
                continue
            candidates.append({**r, "source": "faiss"})
            if len(candidates) >= self._faiss_pool:
                break
        return candidates

    # ------------------------------------------------------------------ #
    # Stage 2b: LLM open-world recall
    # ------------------------------------------------------------------ #

    def _recall_open_world(
        self, query: str, seeker_block: str, history_imdb_ids: list[str]
    ) -> tuple[list[dict], str]:
        user_msg = f"{seeker_block} {query}".strip() if seeker_block else query
        try:
            raw = self._llm.complete(
                [
                    {"role": "system", "content": OW_RECALL_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=12288,
            )
        except Exception as e:
            return [], f"<error: {e}>"

        parsed = _parse_json_loose(raw)
        if not isinstance(parsed, list):
            if isinstance(parsed, dict):
                parsed = [parsed]
            else:
                return [], raw

        seen = set(history_imdb_ids)
        candidates: list[dict] = []
        seen_ids: set[str] = set()
        for item in parsed:
            if not isinstance(item, dict):
                continue
            title = item.get("title", "").strip()
            if not title:
                continue
            mid = self._resolve_title(title)
            if not mid or mid in seen or mid in seen_ids:
                continue
            movie = get_movie_by_id(mid) or {}
            seen_ids.add(mid)
            candidates.append(
                {
                    "id": mid,
                    "imdb_id": mid,
                    "title": movie.get("title", title),
                    "year": movie.get("year", item.get("year", "")),
                    "genres": movie.get("genres", []),
                    "overview": movie.get("overview", ""),
                    "source": "llm",
                }
            )
            if len(candidates) >= self._llm_recall:
                break
        return candidates, raw

    def _resolve_title(self, title: str) -> str | None:
        result = self._index.search(title, top_k=5)
        if result["exact_canonical_match"]:
            m = result["exact_canonical_match"][0]
        elif result["bm25"]:
            m = result["bm25"][0]
        else:
            return None
        return m.get("imdb_id") or m.get("local_movie_id")

    # ------------------------------------------------------------------ #
    # Stage 3: merge and dedup
    # ------------------------------------------------------------------ #

    def _merge_candidates(
        self,
        faiss: list[dict],
        llm: list[dict],
        history_imdb_ids: list[str],
    ) -> list[dict]:
        seen = set(history_imdb_ids)
        merged: list[dict] = []
        for c in faiss + llm:
            mid = c.get("imdb_id") or c["id"]
            if mid and mid not in seen:
                seen.add(mid)
                merged.append(c)
        return merged

    # ------------------------------------------------------------------ #
    # Stage 4: rerank
    # ------------------------------------------------------------------ #

    def _rerank(
        self,
        prefs: dict,
        candidates: list[dict],
        top_k: int,
        seeker_block: str = "",
        query: str = "",
    ) -> tuple[list[int], str]:
        lines = []
        for i, c in enumerate(candidates):
            title = c.get("title", "")
            year = c.get("year", "")
            genres = ", ".join(c.get("genres", [])[:3])
            overview = (c.get("overview") or "")[:160]
            source = c.get("source", "")
            line = f"[{i}|{source}] {title}"
            if year:
                line += f" ({year})"
            if genres:
                line += f" — {genres}"
            if overview:
                line += f". {overview}"
            lines.append(line)

        prefs_summary = json.dumps(
            {k: prefs.get(k, []) for k in ("genres", "moods", "likes", "dislikes")},
            ensure_ascii=False,
        )
        context_parts = []
        if seeker_block:
            context_parts.append(f"Recent conversation:\n{seeker_block}")
        if query:
            context_parts.append(f"Annotated user query:\n{query}")
        context_parts.append(f"Extracted preferences:\n{prefs_summary}")
        context_section = "\n\n".join(context_parts)

        user_msg = (
            f"{context_section}\n\n"
            f"Candidates:\n" + "\n".join(lines) + "\n\n"
            f'Return the {top_k} best indices as {{"ranking": [0, 3, 7, ...]}}.'
        )

        try:
            raw = self._llm.complete(
                [
                    {"role": "system", "content": RERANK_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=12288,
            )
        except Exception as e:
            return [], f"<error: {e}>"

        parsed = _parse_json_loose(raw)
        ranking: list[int] = []
        if isinstance(parsed, dict):
            arr = parsed.get("ranking", [])
            if isinstance(arr, list):
                for x in arr:
                    try:
                        idx = int(x)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= idx < len(candidates) and idx not in ranking:
                        ranking.append(idx)
        return ranking, raw

    def _materialize_picks(
        self, ranking: list[int], candidates: list[dict], top_k: int
    ) -> list[str]:
        used: set[int] = set(ranking)
        ordered = list(ranking)
        # Pad from candidates in original order if rerank returned fewer than top_k
        if len(ordered) < top_k:
            for i in range(len(candidates)):
                if i not in used:
                    ordered.append(i)
                if len(ordered) >= top_k:
                    break

        picks: list[str] = []
        seen_ids: set[str] = set()
        for i in ordered[:top_k]:
            c = candidates[i]
            mid = c.get("imdb_id") or c["id"]
            if mid and mid not in seen_ids:
                seen_ids.add(mid)
                picks.append(mid)
        return picks

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _format_seeker_block(self, dialog_history: list[Turn]) -> str:
        turns = [
            t.text for t in dialog_history[-8:] if t.role == "seeker" and t.text.strip()
        ]
        return " ".join(turns[-3:])

    def _format_history_block(self, history_imdb_ids: list[str]) -> str:
        lines = []
        for mid in history_imdb_ids:
            movie = get_movie_by_id(mid)
            if not movie:
                continue
            title = movie.get("title", "")
            year = movie.get("year", "")
            genres = ", ".join(movie.get("genres", [])[:3])
            label = f"- {title}"
            if year:
                label += f" ({year})"
            if genres:
                label += f" [{genres}]"
            lines.append(label)
        return "\n".join(lines)
