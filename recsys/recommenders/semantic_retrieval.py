from recsys.data.inspired import Turn
from recsys.recommenders.base import BaseRecommender, RecommendationResult
from recsys.retrieval.searcher import Searcher


class SemanticRetrievalRecommender(BaseRecommender):
    name = "semantic_retrieval"

    def __init__(self, searcher: Searcher | None = None):
        self._searcher = searcher

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
        seeker_turns = [t.text for t in dialog_history[-6:] if t.role == "seeker" and t.text.strip()]
        context = " ".join(seeker_turns[-2:])
        enriched = f"{context} {query}".strip() if context else query

        seen = set(history_imdb_ids)
        results = self._get_searcher().search(enriched, top_k=top_k * 2)
        picks = []
        for r in results:
            mid = r.get("imdb_id") or r["id"]
            if mid in seen:
                continue
            picks.append(mid)
            if len(picks) >= top_k:
                break

        return RecommendationResult(
            movie_ids=picks,
            explanation=f"Semantic similarity to: {enriched[:120]}{'...' if len(enriched) > 120 else ''}",
        )
