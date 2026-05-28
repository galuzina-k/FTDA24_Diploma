import json

import numpy as np

from recsys.config import EMBEDDING_MODEL, FAISS_IDS_PATH, FAISS_INDEX_PATH, TOP_K
from recsys.data.catalog import load_catalog
from recsys.retrieval.build_index import build_index

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _needs_bge_prefix(model_name: str) -> bool:
    name = model_name.lower()
    return "bge" in name and "v1.5" in name


class Searcher:
    def __init__(self):
        import faiss
        from sentence_transformers import SentenceTransformer

        if not FAISS_INDEX_PATH.exists():
            build_index()

        self._faiss = faiss
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        self.ids = json.loads(FAISS_IDS_PATH.read_text())
        self.catalog_by_id = {m["id"]: m for m in load_catalog()}
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self._query_prefix = (
            BGE_QUERY_PREFIX if _needs_bge_prefix(EMBEDDING_MODEL) else ""
        )

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        text = self._query_prefix + query if self._query_prefix else query
        emb = self.model.encode([text], convert_to_numpy=True).astype(np.float32)
        self._faiss.normalize_L2(emb)
        scores, indices = self.index.search(emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            movie_id = self.ids[idx]
            movie = self.catalog_by_id.get(
                movie_id, {"id": movie_id, "title": "Unknown"}
            )
            results.append({**movie, "score": float(score)})
        return results
