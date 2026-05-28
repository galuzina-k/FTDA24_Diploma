import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from recsys.config import RECOMMENDER_LOGS_DIR, TOP_K
from recsys.data.inspired import Dialog, dialog_context_before_recommendation
from recsys.evaluation.metrics import compute_metrics
from recsys.recommenders.base import BaseRecommender


def _log_path(recommender_name: str) -> Path:
    return RECOMMENDER_LOGS_DIR / f"{recommender_name}.jsonl"


def _load_done(log_path: Path) -> dict[str, list[str]]:
    if not log_path.exists():
        return {}
    done = {}
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                done[entry["dialog_id"]] = entry["predictions"]
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def _predict_one(recommender: BaseRecommender, dialog: Dialog, top_k: int) -> dict:
    ctx = dialog_context_before_recommendation(dialog)
    result = recommender.recommend(
        ctx, dialog.user_query, dialog.history_imdb_ids, top_k=top_k
    )
    return {
        "dialog_id": dialog.dialog_id,
        "target": dialog.target_imdb_id,
        "predictions": result.movie_ids,
        "extra": result.extra,
    }


def evaluate_recommender(
    recommender: BaseRecommender,
    train_dialogs: list[Dialog],
    test_dialogs: list[Dialog],
    top_k: int = TOP_K,
    workers: int = 1,
    resume: bool = True,
) -> dict:
    RECOMMENDER_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _log_path(recommender.name)

    recommender.fit(train_dialogs)

    done = _load_done(log_path) if resume else {}
    remaining = [d for d in test_dialogs if d.dialog_id not in done]
    if done:
        print(f"  Resuming: {len(done)} cached, {len(remaining)} remaining")

    write_lock = threading.Lock()
    out = open(log_path, "a", encoding="utf-8")

    def write(entry: dict):
        with write_lock:
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            out.flush()

    try:
        if workers <= 1:
            for d in tqdm(remaining, desc=f"  {recommender.name}"):
                entry = _predict_one(recommender, d, top_k)
                done[d.dialog_id] = entry["predictions"]
                write(entry)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_predict_one, recommender, d, top_k): d
                    for d in remaining
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"  {recommender.name}",
                ):
                    d = futures[future]
                    try:
                        entry = future.result()
                        done[d.dialog_id] = entry["predictions"]
                        write(entry)
                    except Exception as e:
                        print(f"\n  FAILED {d.dialog_id}: {e}")
    finally:
        out.close()

    predictions, targets = [], []
    for d in test_dialogs:
        if d.dialog_id in done:
            predictions.append(done[d.dialog_id])
            targets.append([d.target_imdb_id])

    metrics = compute_metrics(predictions, targets, k=top_k)
    metrics["recommender"] = recommender.name
    return metrics
