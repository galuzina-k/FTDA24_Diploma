import json

from recsys.data.inspired import Turn
from recsys.recommenders.base import RecommendationResult
from recsys.recommenders.hybrid import HybridRecommender, RERANK_PROMPT, _parse_json_loose


STATE_UPDATE_PROMPT = """\
You are tracking the state of a movie-recommendation conversation, turn by turn.

You will receive:
- The current state (JSON object with `seen_titles`, `rejected_titles`, `constraints`)
- The most recent recommender turn (what the bot just said)
- The most recent seeker turn (what the user said in response)

Update the state and return a JSON object with the same shape:
- `seen_titles`: movies the seeker UNAMBIGUOUSLY states they have ALREADY watched in the past (completed past viewing).
- `rejected_titles`: movies the RECOMMENDER just pitched that the seeker CLEARLY rejected (will not watch / does not want).
- `constraints`: short strings describing situational viewing constraints the seeker has expressed.

Rules:
- Carry over all existing state items; only ADD new ones, never remove.
- Titles should be the canonical movie title (e.g. "The Matrix"). One title per entry — never a series name like "Bond movies".

`seen_titles` — be strict. ONLY add a title if the seeker uses unambiguous past-tense viewing language about THAT specific title:
  YES: "I saw X", "I've watched X", "I watched X last year", "I've seen X", "we rewatched X"
  NO:  "I want to see X", "I'm planning to watch X", "we're getting X", "I'm going to see X",
       "I'd watch X", "I might see X", "X is on my list", "I heard about X", "X looks good",
       "my fiance loves X" (that's someone else), "I love X" (loving ≠ watching)
  If in doubt, DO NOT add the title.

`rejected_titles` — be strict. ONLY add a title that the RECOMMENDER actually pitched in this exchange AND the seeker's reply is clearly negative about it:
  YES: "I didn't like X", "not really into X", "X was boring", "no thanks, something else"
  NO:  "I haven't seen X" (neutral), "I don't know X" (neutral), silence about a pitched title
  Do NOT mark a title as rejected just because the seeker asks for "something else" without naming what they're rejecting.

`constraints` — ONLY situational/contextual factors (who, when, where, runtime, mood-for-tonight, MPAA-style guardrails).
  YES: "watching with partner", "family-friendly", "date night", "short runtime", "no graphic violence", "Sunday afternoon"
  NO:  genre preferences ("likes action"), actor preferences, mood adjectives describing the desired film ("dark", "fun")
       — those are preferences and are tracked elsewhere.

Output ONLY a raw JSON object with the three fields. No preamble, no explanation."""


class HybridStateRecommender(HybridRecommender):
    """Hybrid + turn-by-turn conversation state.

    On top of HybridRecommender's preference extraction, this builds an
    incrementally-updated ConversationState across the seeker turns:

      - rejected_titles: HARD filter — drop matching IMDb IDs from the pool.
        Principled: a title only lands here if the bot pitched it AND the
        seeker said no. False positives are unlikely.
      - seen_titles:     SOFT signal — surfaced in the rerank prompt so the
        LLM can deprioritize them. Used to be a hard filter, but the state
        extractor was too eager to mark titles as "seen" (interpreting
        "I want to see X" as past viewing), blocking valid candidates.
      - constraints:     SOFT signal — surfaced in the rerank prompt.

    The state-update LLM call runs once per seeker turn (after each recommender
    response), so cost is O(N_seeker_turns) extra LLM calls per dialog. Unlike
    `hybrid`, the seen/rejected lists structurally exist — one-shot extraction
    cannot reliably produce them.
    """

    name = "hybrid_state"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        slug = self._llm.model.replace("/", "_").replace("-", "_").replace(".", "_")
        self.name = f"hybrid_state__{slug}"

    # ------------------------------------------------------------------ #
    # Override recommend() to thread state through the pipeline
    # ------------------------------------------------------------------ #

    def recommend(
        self,
        dialog_history: list[Turn],
        query: str,
        history_imdb_ids: list[str],
        top_k: int = 10,
    ) -> RecommendationResult:
        state = self._build_state(dialog_history)
        seen_ids = self._resolve_titles_to_ids(state.get("seen_titles", []))
        rejected_ids = self._resolve_titles_to_ids(state.get("rejected_titles", []))
        # Only rejected_titles is a hard pool filter. seen_titles is surfaced
        # as a soft signal in the rerank prompt — too risky to filter on,
        # because the extractor can mistake "want to see X" for past viewing.
        filter_ids = set(history_imdb_ids) | rejected_ids

        seeker_block = self._format_seeker_block(dialog_history)
        history_block = self._format_history_block(history_imdb_ids)

        prefs, prefs_raw = self._extract_preferences(query, seeker_block, history_block)

        search_text = self._build_search_text(query, seeker_block, prefs)
        faiss_candidates = self._retrieve_faiss(search_text, list(filter_ids))

        llm_candidates, ow_raw = self._recall_open_world(query, seeker_block, list(filter_ids))

        candidates = self._merge_candidates(faiss_candidates, llm_candidates, list(filter_ids))

        if not candidates:
            return RecommendationResult(
                movie_ids=[],
                explanation="No candidates after state filtering.",
                extra={
                    "state": state,
                    "filtered_ids": sorted(rejected_ids),
                    "seen_ids_soft": sorted(seen_ids),
                    "preferences": prefs,
                    "preferences_raw": prefs_raw,
                    "ow_raw": ow_raw,
                },
            )

        seen_titles_resolved = self._titles_for_ids(seen_ids, state.get("seen_titles", []))
        ranking, rerank_raw = self._rerank_with_constraints(
            prefs, candidates, top_k, seeker_block, query,
            state.get("constraints", []), seen_titles_resolved,
        )
        picks = self._materialize_picks(ranking, candidates, top_k)

        n_faiss = sum(1 for c in candidates if c["source"] == "faiss")
        n_llm = sum(1 for c in candidates if c["source"] == "llm")
        return RecommendationResult(
            movie_ids=picks,
            explanation=(
                f"Hybrid+state: hard-filtered {len(rejected_ids)} rejected ids "
                f"(+{len(seen_ids)} seen as soft signal), "
                f"FAISS ({n_faiss}) + LLM recall ({n_llm}) = {len(candidates)} candidates "
                f"→ rerank → top {top_k}."
            ),
            extra={
                "state": state,
                "filtered_ids": sorted(rejected_ids),
                "seen_ids_soft": sorted(seen_ids),
                "preferences": prefs,
                "preferences_raw": prefs_raw,
                "ow_raw": ow_raw,
                "candidates": [{"id": c.get("imdb_id") or c["id"], "source": c["source"]} for c in candidates],
                "rerank_raw": rerank_raw,
                "ranking": ranking,
            },
        )

    # ------------------------------------------------------------------ #
    # State tracking
    # ------------------------------------------------------------------ #

    def _build_state(self, dialog_history: list[Turn]) -> dict:
        """Walk the dialog turn by turn, calling the state-update LLM on each
        recommender→seeker exchange.
        """
        state: dict = {"seen_titles": [], "rejected_titles": [], "constraints": []}
        last_rec_text = ""
        for turn in dialog_history:
            if turn.role == "recommender":
                last_rec_text = turn.text
                continue
            if turn.role != "seeker" or not turn.text.strip():
                continue
            state = self._update_state(state, last_rec_text, turn.text)
            last_rec_text = ""
        return state

    def _update_state(self, state: dict, recommender_text: str, seeker_text: str) -> dict:
        user_msg = (
            f"Current state:\n{json.dumps(state, ensure_ascii=False)}\n\n"
            f"Recommender turn:\n{recommender_text or '(none)'}\n\n"
            f"Seeker turn:\n{seeker_text}"
        )
        try:
            raw = self._llm.complete(
                [
                    {"role": "system", "content": STATE_UPDATE_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=512,
            )
        except Exception:
            return state

        parsed = _parse_json_loose(raw)
        if not isinstance(parsed, dict):
            return state

        return {
            "seen_titles": _merge_str_list(state.get("seen_titles", []), parsed.get("seen_titles")),
            "rejected_titles": _merge_str_list(state.get("rejected_titles", []), parsed.get("rejected_titles")),
            "constraints": _merge_str_list(state.get("constraints", []), parsed.get("constraints")),
        }

    def _resolve_titles_to_ids(self, titles: list[str]) -> set[str]:
        ids: set[str] = set()
        for title in titles:
            if not isinstance(title, str) or not title.strip():
                continue
            mid = self._resolve_title(title)
            if mid:
                ids.add(mid)
        return ids

    def _titles_for_ids(self, ids: set[str], fallback_titles: list[str]) -> list[str]:
        """Return human-readable titles for the seen-soft signal in the rerank
        prompt. Prefers canonical catalog titles when an IMDb id resolved;
        falls back to the raw extracted titles otherwise.
        """
        from recsys.data.catalog import get_movie_by_id
        out: list[str] = []
        seen_lower: set[str] = set()
        for mid in ids:
            m = get_movie_by_id(mid) or {}
            t = m.get("title")
            if t and t.lower() not in seen_lower:
                seen_lower.add(t.lower())
                out.append(t)
        for t in fallback_titles or []:
            if isinstance(t, str) and t.strip() and t.lower() not in seen_lower:
                seen_lower.add(t.lower())
                out.append(t.strip())
        return out

    # ------------------------------------------------------------------ #
    # Rerank with constraints injected as a soft signal
    # ------------------------------------------------------------------ #

    def _rerank_with_constraints(
        self,
        prefs: dict,
        candidates: list[dict],
        top_k: int,
        seeker_block: str,
        query: str,
        constraints: list[str],
        seen_titles: list[str],
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
        if constraints:
            context_parts.append(f"Situational constraints:\n- " + "\n- ".join(constraints))
        if seen_titles:
            context_parts.append(
                "Movies the seeker has already watched (deprioritize unless clearly the best fit):\n- "
                + "\n- ".join(seen_titles)
            )
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


def _merge_str_list(existing: list, new) -> list[str]:
    """Carry-over union: existing items kept, new ones appended, case-insensitive dedup."""
    out: list[str] = []
    seen_lower: set[str] = set()
    for item in list(existing or []) + (list(new) if isinstance(new, list) else []):
        if not isinstance(item, str):
            continue
        s = item.strip()
        if not s:
            continue
        key = s.lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        out.append(s)
    return out
