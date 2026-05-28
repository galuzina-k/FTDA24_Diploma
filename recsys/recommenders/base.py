from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from recsys.data.inspired import Dialog, Turn


@dataclass
class RecommendationResult:
    movie_ids: list[str]
    explanation: str = ""
    extra: dict = field(default_factory=dict)


class BaseRecommender(ABC):
    name: str = "base"

    def fit(self, train_dialogs: list[Dialog]) -> None:
        pass

    @abstractmethod
    def recommend(
        self,
        dialog_history: list[Turn],
        query: str,
        history_imdb_ids: list[str],
        top_k: int = 10,
    ) -> RecommendationResult: ...
