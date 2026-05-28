import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recsys.config import EMBEDDING_MODEL, FAISS_INDEX_PATH
from recsys.retrieval.build_index import build_index


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from catalog.")
    parser.add_argument(
        "--model",
        default=None,
        help=f"Embedding model name (default: {EMBEDDING_MODEL} from config/.env).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the index already exists.",
    )
    args = parser.parse_args()

    if FAISS_INDEX_PATH.exists() and not args.force:
        print(f"Index already exists at {FAISS_INDEX_PATH}. Use --force to rebuild.")
        return

    if args.model:
        import recsys.config as cfg

        cfg.EMBEDDING_MODEL = args.model

    build_index()


if __name__ == "__main__":
    main()
