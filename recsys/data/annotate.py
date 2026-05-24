import csv
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from recsys.config import (
    INSPIRED_ANNOTATED_PATH,
    INSPIRED_DEV_TSV,
    INSPIRED_TEST_TSV,
    INSPIRED_TRAIN_TSV,
)
from recsys.llm.client import LLMClient

WORKERS = 20

SYSTEM_PROMPT = """You are given a movie recommendation dialogue between a SEEKER and a RECOMMENDER.
Extract the following information and return a JSON object with exactly these fields:

{
  "user_query": "A single natural language query that in details summarize the seeker's preferences and what kind of movie they want. Include seeker's opinion on films/genres/directors (mention all the names).",
  "recommended_movie": "The movie finally recommended and accepted by the seeker, or empty string if none",
  "movies": {"Movie Title": "good|bad|neutral", ...},
  "genres": {"genre name": "good|bad|neutral", ...},
  "actors": {"actor name": "good|bad|neutral", ...},
  "directors": {"director name": "good|bad|neutral", ...},
  "others": {"person name": "good|bad|neutral", ...}
}

Rules:
- user_query/movies/actors/directors/others must NOT include anything related to the finally recommended movie
- sentiment (good/bad/neutral) reflects how the seeker feels about each item
- return only the JSON, no extra text
"""


def load_all_dialogs() -> dict[str, list[dict]]:
    dialogs: dict[str, list[dict]] = {}
    for path in (INSPIRED_TRAIN_TSV, INSPIRED_DEV_TSV, INSPIRED_TEST_TSV):
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                dialogs.setdefault(row["dialog_id"], []).append(row)
    return dialogs


def _build_full_text(rows: list[dict]) -> str:
    return "\n".join(f"{r['speaker']}: {r['text']}" for r in rows)


def _get_labels(rows: list[dict]) -> tuple[str, str]:
    for row in reversed(rows):
        coarse = row.get("coarse_label", "").strip()
        fine = row.get("fine_label", "").strip()
        if coarse or fine:
            return coarse, fine
    return "", ""


def annotate_dialog(llm: LLMClient, dialog_id: str, rows: list[dict]) -> dict:
    full_text = _build_full_text(rows)
    coarse, fine = _get_labels(rows)
    extracted = llm.complete_json(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": full_text}],
        temperature=0.0, max_tokens=1024,
    )
    return {
        "dialog_id": dialog_id,
        "full_text": full_text,
        "coarse_label": coarse,
        "fine_label": fine,
        **extracted,
    }


def annotate_all(output_path: Path = INSPIRED_ANNOTATED_PATH) -> None:
    llm = LLMClient()
    dialogs = load_all_dialogs()

    done = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                done.add(json.loads(line)["dialog_id"])
        print(f"Resuming — {len(done)} done, {len(dialogs) - len(done)} remaining")

    remaining = [(did, rows) for did, rows in dialogs.items() if did not in done]
    write_lock = threading.Lock()
    failed = []

    with open(output_path, "a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {executor.submit(annotate_dialog, llm, did, rows): did for did, rows in remaining}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Annotating"):
                did = futures[future]
                try:
                    result = future.result()
                    with write_lock:
                        out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        out.flush()
                except Exception as e:
                    print(f"FAILED {did}: {e}")
                    failed.append(did)

    print(f"Done → {output_path}")
    if failed:
        print(f"Failed: {len(failed)}: {failed[:5]}")
