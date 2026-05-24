import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from recsys.config import INSPIRED_ANNOTATED_PATH, MOVIE_ID_MAP_PATH
from recsys.data.title_index import TitleIndex
from recsys.llm.client import LLMClient

LLM_WORKERS = 10

LLM_PROMPT = """\
A movie is referenced in a dialogue as: "{mention}"

Here are the top candidates from a movie database:
{candidates}

Which candidate best matches the reference? Reply with just the exact title from the list, or "none" if none match."""


def _format_candidates(candidates: list[dict]) -> str:
    return "\n".join(f"- {c['title']} ({c['year']}) [id: {c['local_movie_id']}]" for c in candidates)


def resolve_mention(mention: str, index: TitleIndex, llm: LLMClient | None) -> str | None:
    result = index.search(mention, top_k=5)

    if result["exact_canonical_match"]:
        m = result["exact_canonical_match"][0]
        return m["imdb_id"] or m["local_movie_id"]

    candidates = result["bm25"][:5]
    if not candidates:
        return None

    if llm is None:
        return candidates[0]["imdb_id"] or candidates[0]["local_movie_id"]

    prompt = LLM_PROMPT.format(mention=mention, candidates=_format_candidates(candidates))
    response = llm.complete([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=64).strip()

    if response.lower() == "none":
        return None

    response_lower = response.lower()
    for c in candidates:
        if c["title"].lower() in response_lower or response_lower in c["title"].lower():
            return c["imdb_id"] or c["local_movie_id"]

    return candidates[0]["imdb_id"] or candidates[0]["local_movie_id"]


def _collect_mentions(annotated_path: Path) -> dict[str, list[str]]:
    result = {}
    with open(annotated_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            mentions = list(d.get("movies", {}).keys())
            if d.get("recommended_movie"):
                mentions = [d["recommended_movie"]] + mentions
            seen, unique = set(), []
            for m in mentions:
                if m and m not in seen:
                    seen.add(m)
                    unique.append(m)
            result[d["dialog_id"]] = unique
    return result


def build_id_map(
    annotated_path: Path = INSPIRED_ANNOTATED_PATH,
    output_path: Path = MOVIE_ID_MAP_PATH,
) -> None:
    index = TitleIndex()
    llm = LLMClient()

    dialog_mentions = _collect_mentions(annotated_path)
    unique_mentions = list({m for ms in dialog_mentions.values() for m in ms})
    print(f"{len(dialog_mentions)} dialogs, {len(unique_mentions)} unique mentions")

    cache_path = output_path.parent / "movie_id_map_cache.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mention_to_id: dict[str, str | None] = {}
    if cache_path.exists():
        mention_to_id = json.loads(cache_path.read_text())
        print(f"Resumed from cache: {len(mention_to_id)} already resolved")

    remaining = [m for m in unique_mentions if m not in mention_to_id]
    write_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=LLM_WORKERS) as executor:
        futures = {executor.submit(resolve_mention, m, index, llm): m for m in remaining}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Matching"):
            mention = futures[future]
            try:
                movie_id = future.result(timeout=30)
            except Exception as e:
                print(f"\nFAILED {mention!r}: {e}")
                movie_id = None
            with write_lock:
                mention_to_id[mention] = movie_id
                cache_path.write_text(json.dumps(mention_to_id, ensure_ascii=False))

    output = {
        dialog_id: {m: mention_to_id.get(m) for m in mentions}
        for dialog_id, mentions in dialog_mentions.items()
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))

    matched = sum(1 for v in mention_to_id.values() if v)
    print(f"Matched {matched}/{len(mention_to_id)} ({100 * matched // len(mention_to_id)}%) → {output_path}")
