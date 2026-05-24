import csv
import json
from functools import lru_cache
from pathlib import Path

from recsys.config import CATALOG_PATH, INSPIRED_MOVIE_DB_PATH


def _detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        sample = f.read(4096)
    return "\t" if sample.count("\t") >= sample.count(",") else ","


def _normalize(raw: dict, idx: int) -> dict:
    title = (raw.get("title") or raw.get("movie_title") or f"Movie_{idx}").strip()
    year = (raw.get("year") or raw.get("release_year") or "").strip()
    genres_raw = raw.get("genre") or raw.get("genres") or ""
    genres = [g.strip() for g in genres_raw.replace("|", ",").split(",") if g.strip()]
    overview = (raw.get("short_plot") or raw.get("long_plot") or raw.get("overview") or "").strip()
    actors = (raw.get("actors") or raw.get("cast") or "").strip()
    director = (raw.get("director") or "").strip()
    movie_id = (raw.get("movie_id") or raw.get("imdb_id") or str(idx)).strip()
    imdb_id = (raw.get("imdb_id") or "").strip()

    text_parts = [f"{title} ({year})" if year else title]
    if genres:
        text_parts.append("Genres: " + ", ".join(genres))
    if director:
        text_parts.append(f"Director: {director}")
    if actors:
        text_parts.append("Cast: " + actors)
    if overview:
        text_parts.append(overview)

    return {
        "id": movie_id,
        "imdb_id": imdb_id,
        "title": title,
        "year": year,
        "genres": genres,
        "actors": actors,
        "director": director,
        "overview": overview,
        "text": ". ".join(text_parts),
    }


def build_catalog(
    movie_db_path: Path = INSPIRED_MOVIE_DB_PATH,
    output_path: Path = CATALOG_PATH,
) -> list[dict]:
    delimiter = _detect_delimiter(movie_db_path)

    with movie_db_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        movies = [_normalize(row, i) for i, row in enumerate(reader, start=1)]

    seen, unique = {}, []
    for m in movies:
        key = m["title"].lower().strip()
        if key not in seen:
            seen[key] = True
            unique.append(m)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(unique, ensure_ascii=False, indent=2))
    print(f"Built catalog: {len(unique)} movies → {output_path}")
    return unique


@lru_cache(maxsize=1)
def load_catalog() -> list[dict]:
    if not CATALOG_PATH.exists():
        build_catalog()
    return json.loads(CATALOG_PATH.read_text())


@lru_cache(maxsize=1)
def _by_id() -> dict[str, dict]:
    return {m["id"]: m for m in load_catalog()}


@lru_cache(maxsize=1)
def _by_imdb_id() -> dict[str, dict]:
    return {m["imdb_id"]: m for m in load_catalog() if m["imdb_id"]}


@lru_cache(maxsize=1)
def _by_title_lower() -> dict[str, dict]:
    return {m["title"].lower().strip(): m for m in load_catalog()}


def get_movie_by_id(movie_id: str) -> dict | None:
    return _by_id().get(movie_id) or _by_imdb_id().get(movie_id)


def get_movie_by_title(title: str) -> dict | None:
    key = title.lower().strip()
    movie = _by_title_lower().get(key)
    if movie:
        return movie
    for t, m in _by_title_lower().items():
        if key in t:
            return m
    return None
