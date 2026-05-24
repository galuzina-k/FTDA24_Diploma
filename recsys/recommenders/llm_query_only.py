import json
import re

from recsys.data.inspired import Turn
from recsys.data.title_index import TitleIndex
from recsys.llm.client import LLMClient
from recsys.recommenders.base import BaseRecommender, RecommendationResult

SYSTEM_PROMPT = """\
You are a movie recommendation assistant. The user describes what they are looking for.
Recommend exactly 10 movies that best match their request, ranked from best to worst fit.
Reply with a JSON array of objects, each with "title" and "year" fields.
Example: [{"title": "The Matrix", "year": "1999"}, ...]
No explanation, just the JSON array."""


class LLMQueryOnlyRecommender(BaseRecommender):
    name = "llm_query_only"

    def __init__(self, llm: LLMClient | None = None, title_index: TitleIndex | None = None):
        self._llm = llm or LLMClient()
        self._index = title_index or TitleIndex()
        slug = self._llm.model.replace("/", "_").replace("-", "_").replace(".", "_")
        self.name = f"llm_query_only__{slug}"

    def recommend(
        self,
        dialog_history: list[Turn],
        query: str,
        history_imdb_ids: list[str],
        top_k: int = 10,
    ) -> RecommendationResult:
        if not query:
            return RecommendationResult(movie_ids=[], explanation="No user_query available.")

        raw = self._llm.complete(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}],
            temperature=0.0, max_tokens=2048,
        )
        resolved = self._resolve_titles(raw)

        seen = set(history_imdb_ids)
        picks = [mid for mid in resolved if mid not in seen][:top_k]
        return RecommendationResult(
            movie_ids=picks,
            explanation="LLM open-world recall, resolved to catalog via title index.",
            extra={"llm_response": raw},
        )

    @staticmethod
    def _proxy_query_from_turns(dialog_history: list[Turn]) -> str:
        seeker_texts = [t.text.strip() for t in dialog_history if t.role == "seeker" and t.text.strip()]
        if not seeker_texts:
            return ""
        joined = "\n".join(f"- {t}" for t in seeker_texts)
        return f"Here is what the movie seeker said during the conversation:\n{joined}\n\nBased on this, recommend movies that match their preferences."

    def _resolve_titles(self, llm_response: str) -> list[str]:
        text = llm_response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                data = [data]
        except (json.JSONDecodeError, AttributeError):
            data = []
            for m in re.finditer(r"\{[^{}]+\}", text):
                try:
                    data.append(json.loads(m.group()))
                except json.JSONDecodeError:
                    pass

        resolved, seen = [], set()
        for item in data:
            if not isinstance(item, dict):
                continue
            title = item.get("title", "").strip()
            if not title:
                continue
            result = self._index.search(title, top_k=5)
            if result["exact_canonical_match"]:
                m = result["exact_canonical_match"][0]
            elif result["bm25"]:
                m = result["bm25"][0]
            else:
                continue
            mid = m["imdb_id"] or m["local_movie_id"]
            if mid and mid not in seen:
                resolved.append(mid)
                seen.add(mid)
        return resolved


class LLMQueryOnlyDynamicRecommender(LLMQueryOnlyRecommender):
    """Like LLMQueryOnlyRecommender but falls back to a proxy query built from
    seeker turns when the annotated user_query is unavailable (e.g. at
    intermediate cutoffs in the turn-level evaluation)."""

    name = "llm_query_only_dynamic"

    def __init__(self, llm: LLMClient | None = None, title_index: TitleIndex | None = None):
        super().__init__(llm=llm, title_index=title_index)
        slug = self._llm.model.replace("/", "_").replace("-", "_").replace(".", "_")
        self.name = f"llm_query_only_dynamic__{slug}"

    def recommend(
        self,
        dialog_history: list[Turn],
        query: str,
        history_imdb_ids: list[str],
        top_k: int = 10,
    ) -> RecommendationResult:
        effective_query = query or self._proxy_query_from_turns(dialog_history)
        if not effective_query:
            return RecommendationResult(movie_ids=[], explanation="No context available.")

        raw = self._llm.complete(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": effective_query}],
            temperature=0.0, max_tokens=2048,
        )
        resolved = self._resolve_titles(raw)

        seen = set(history_imdb_ids)
        picks = [mid for mid in resolved if mid not in seen][:top_k]
        return RecommendationResult(
            movie_ids=picks,
            explanation="LLM open-world recall (dynamic query), resolved via title index.",
            extra={"llm_response": raw, "used_proxy_query": not bool(query)},
        )
