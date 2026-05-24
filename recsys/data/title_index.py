import csv
import json
import math
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from tqdm import tqdm

from recsys.config import INSPIRED_MOVIE_DB_PATH, TITLE_INDEX_PATH

NGRAM_RANGE = (3, 5)
TOP_K = 5


def canonicalize_title(title: str) -> str:
    text = unicodedata.normalize("NFKC", title).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[_/\\|]+", " ", text)
    text = re.sub(r"[-:]+", " ", text)
    text = re.sub(r"\bvs\b", "v", text)
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        sample = f.read(4096)
    return "\t" if sample.count("\t") >= sample.count(",") else ","


def _char_ngrams(text: str, min_n: int = NGRAM_RANGE[0], max_n: int = NGRAM_RANGE[1]) -> list[str]:
    padded = f"  {text}  "
    grams = []
    for n in range(min_n, max_n + 1):
        for i in range(0, max(0, len(padded) - n + 1)):
            grams.append(padded[i:i + n])
    return grams


def _seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _token_sort_ratio(a: str, b: str) -> float:
    return _seq_ratio(" ".join(sorted(a.split())), " ".join(sorted(b.split())))


def _token_set_ratio(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    common = " ".join(sorted(sa & sb))
    fa = " ".join(sorted(sa))
    fb = " ".join(sorted(sb))
    return max(_seq_ratio(common, fa), _seq_ratio(common, fb), _seq_ratio(fa, fb))


def _fuzzy_score(query: str, candidate: str) -> float:
    return max(_seq_ratio(query, candidate), _token_sort_ratio(query, candidate), _token_set_ratio(query, candidate))


def build_title_retrieval_index(
    movie_db_path: Path = INSPIRED_MOVIE_DB_PATH,
    index_path: Path = TITLE_INDEX_PATH,
) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    delimiter = _detect_delimiter(movie_db_path)

    movies = []
    df_counter: Counter = Counter()

    with movie_db_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for idx, row in enumerate(tqdm(reader, desc="Building title index"), start=1):
            title = (row.get("title") or row.get("movie_title") or f"movie_{idx}").strip()
            canonical = canonicalize_title(title)
            ngrams = _char_ngrams(canonical)
            df_counter.update(set(ngrams))
            movies.append({
                "local_movie_id": (row.get("movie_id") or row.get("id") or str(idx)).strip(),
                "title": title,
                "canonical_title": canonical,
                "year": (row.get("year") or "").strip(),
                "imdb_id": (row.get("imdb_id") or "").strip(),
                "ngrams": dict(Counter(ngrams)),
            })

    total = len(movies)
    idf = {gram: math.log((1 + total) / (1 + df)) + 1.0 for gram, df in df_counter.items()}

    with index_path.open("w", encoding="utf-8") as f:
        json.dump({"idf": idf, "movies": movies}, f, ensure_ascii=False)


class TitleIndex:
    def __init__(self, index_path: Path = TITLE_INDEX_PATH):
        if not index_path.exists():
            build_title_retrieval_index(index_path=index_path)
        with index_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        self.idf: dict[str, float] = payload["idf"]
        self.movies: list[dict[str, Any]] = payload["movies"]
        for m in self.movies:
            m["doc_len"] = sum(m["ngrams"].values())
        self.avg_len = sum(m["doc_len"] for m in self.movies) / max(len(self.movies), 1)

    def _bm25_score(self, query_grams: list[str], movie: dict) -> float:
        if not query_grams or self.avg_len == 0.0:
            return 0.0
        k1, b = 1.5, 0.75
        doc_counts = movie["ngrams"]
        doc_len = movie["doc_len"]
        score = 0.0
        for term in set(query_grams):
            tf = doc_counts.get(term, 0)
            if tf == 0:
                continue
            score += self.idf.get(term, 0.0) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / self.avg_len))
        return score

    def search(self, mention: str, top_k: int = TOP_K) -> dict[str, Any]:
        canonical = canonicalize_title(mention)
        query_grams = _char_ngrams(canonical)

        exact = [self._meta(m, 1.0) for m in self.movies if m["canonical_title"] == canonical]

        fuzzy = sorted(
            (self._meta(m, round(_fuzzy_score(canonical, m["canonical_title"]), 4)) for m in self.movies),
            key=lambda x: x["score"], reverse=True,
        )[:top_k]

        bm25 = sorted(
            (self._meta(m, round(self._bm25_score(query_grams, m), 4)) for m in self.movies),
            key=lambda x: x["score"], reverse=True,
        )[:top_k]

        return {
            "query": mention,
            "canonical_query": canonical,
            "exact_canonical_match": exact[:top_k],
            "fuzzy_string_matching": fuzzy,
            "bm25": bm25,
        }

    @staticmethod
    def _meta(movie: dict, score: float) -> dict:
        return {
            "local_movie_id": movie["local_movie_id"],
            "title": movie["title"],
            "canonical_title": movie["canonical_title"],
            "year": movie["year"],
            "imdb_id": movie["imdb_id"],
            "score": score,
        }
