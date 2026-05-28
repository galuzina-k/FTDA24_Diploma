import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

INSPIRED_MOVIE_DB_PATH = RAW_DIR / "movie_database.tsv"
INSPIRED_ANNOTATED_PATH = RAW_DIR / "annotated.jsonl"
INSPIRED_TRAIN_TSV = RAW_DIR / "train.tsv"
INSPIRED_DEV_TSV = RAW_DIR / "dev.tsv"
INSPIRED_TEST_TSV = RAW_DIR / "test.tsv"

MOVIE_ID_MAP_PATH = PROCESSED_DIR / "movie_id_map.json"
TITLE_INDEX_PATH = PROCESSED_DIR / "movie_title_retrieval_index.json"
TITLE_METADATA_PATH = PROCESSED_DIR / "movie_database_title_metadata.jsonl"
LLM_BASELINE_LOG_PATH = PROCESSED_DIR / "llm_baseline_logs.jsonl"

CATALOG_PATH = PROCESSED_DIR / "catalog.json"

RETRIEVAL_INDEX_DIR = PROCESSED_DIR / "faiss_index"
FAISS_INDEX_PATH = RETRIEVAL_INDEX_DIR / "movies.faiss"
FAISS_IDS_PATH = RETRIEVAL_INDEX_DIR / "ids.json"

RESULTS_DIR = ROOT_DIR / "results"
RECOMMENDER_LOGS_DIR = RESULTS_DIR / "logs"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "openai/gpt-5.3-chat"  # "minimax/minimax-m2.5"#

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_BASE_URL = "https://api.themoviedb.org/3"

TOP_K = 10
SPLIT_SEED = 42
SPLIT_TEST_RATIO = 0.5
