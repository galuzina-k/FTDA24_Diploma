from collections import Counter

from recsys.data.inspired import Dialog, Turn
from recsys.recommenders.base import BaseRecommender, RecommendationResult


class PopularityRecommender(BaseRecommender):
    name = "popularity"

    def fit(self, train_dialogs: list[Dialog]) -> None:
        counter: Counter = Counter()
        for d in train_dialogs:
            counter[d.target_imdb_id] += 1
            for mid in d.history_imdb_ids:
                counter[mid] += 1
        self.ranked = [mid for mid, _ in counter.most_common()]

    def recommend(
        self,
        dialog_history: list[Turn],
        query: str,
        history_imdb_ids: list[str],
        top_k: int = 10,
    ) -> RecommendationResult:
        seen = set(history_imdb_ids)
        picks = [mid for mid in self.ranked if mid not in seen][:top_k]
        return RecommendationResult(movie_ids=picks, explanation="Globally most popular movies.")
