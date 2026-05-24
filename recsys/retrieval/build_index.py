import json

import numpy as np

from recsys.config import EMBEDDING_MODEL, FAISS_IDS_PATH, FAISS_INDEX_PATH, RETRIEVAL_INDEX_DIR
from recsys.data.catalog import load_catalog


def build_index() -> None:
    import faiss
    from sentence_transformers import SentenceTransformer

    catalog = load_catalog()
    texts = [m["text"] for m in catalog]
    ids = [m["id"] for m in catalog]

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Embedding {len(texts)} movies...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    RETRIEVAL_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    FAISS_IDS_PATH.write_text(json.dumps(ids))

    print(f"Index built: {len(texts)} vectors, dim={embeddings.shape[1]} → {FAISS_INDEX_PATH}")
