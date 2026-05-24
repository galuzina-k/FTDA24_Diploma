from recsys.recommenders.base import BaseRecommender, RecommendationResult
from recsys.recommenders.history_only import HistoryOnlyRecommender
from recsys.recommenders.hybrid import HybridRecommender
from recsys.recommenders.hybrid_state import HybridStateRecommender
from recsys.recommenders.hybrid_unbiased import HybridUnbiasedRecommender
from recsys.recommenders.llm_query_only import LLMQueryOnlyDynamicRecommender, LLMQueryOnlyRecommender
from recsys.recommenders.popularity import PopularityRecommender
from recsys.recommenders.semantic_retrieval import SemanticRetrievalRecommender
from recsys.recommenders.svd import SVDRecommender

REGISTRY: dict[str, type[BaseRecommender]] = {
    PopularityRecommender.name: PopularityRecommender,
    SVDRecommender.name: SVDRecommender,
    HistoryOnlyRecommender.name: HistoryOnlyRecommender,
    SemanticRetrievalRecommender.name: SemanticRetrievalRecommender,
    LLMQueryOnlyRecommender.name: LLMQueryOnlyRecommender,
    LLMQueryOnlyDynamicRecommender.name: LLMQueryOnlyDynamicRecommender,
    "hybrid": HybridRecommender,
    "hybrid_state": HybridStateRecommender,
}


def build(name: str) -> BaseRecommender:
    if name not in REGISTRY:
        raise ValueError(f"Unknown recommender: {name}. Available: {list(REGISTRY)}")
    return REGISTRY[name]()
