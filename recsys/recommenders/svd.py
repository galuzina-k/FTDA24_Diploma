import numpy as np

from recsys.data.inspired import Dialog, Turn
from recsys.recommenders.base import BaseRecommender, RecommendationResult


class SVDRecommender(BaseRecommender):
    name = "svd"

    def __init__(self, n_factors: int = 50):
        self.n_factors = n_factors

    def fit(self, train_dialogs: list[Dialog]) -> None:
        users = [d.dialog_id for d in train_dialogs]
        items = sorted({mid for d in train_dialogs for mid in d.history_imdb_ids + [d.target_imdb_id]})

        self.user_idx = {u: i for i, u in enumerate(users)}
        self.item_idx = {it: i for i, it in enumerate(items)}
        self.items = items

        R = np.zeros((len(users), len(items)), dtype=np.float32)
        for d in train_dialogs:
            u = self.user_idx[d.dialog_id]
            for mid in d.history_imdb_ids:
                R[u, self.item_idx[mid]] = 1.0
            R[u, self.item_idx[d.target_imdb_id]] = 1.0

        k = min(self.n_factors, R.shape[0] - 1, R.shape[1] - 1)
        U, s, Vt = np.linalg.svd(R, full_matrices=False)
        self.item_factors = Vt[:k, :].T

    def recommend(
        self,
        dialog_history: list[Turn],
        query: str,
        history_imdb_ids: list[str],
        top_k: int = 10,
    ) -> RecommendationResult:
        seen = set(history_imdb_ids)
        vecs = [self.item_factors[self.item_idx[mid]] for mid in history_imdb_ids if mid in self.item_idx]
        if not vecs:
            return RecommendationResult(movie_ids=[], explanation="No history items in training vocabulary.")

        user_vec = np.mean(vecs, axis=0)
        scores = {mid: float(user_vec @ self.item_factors[i]) for mid, i in self.item_idx.items() if mid not in seen}
        picks = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return RecommendationResult(movie_ids=picks, explanation="SVD factorization of dialog-item matrix.")
