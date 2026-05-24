import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recsys.config import RESULTS_DIR, TOP_K
from recsys.data.inspired import load_dialogs
from recsys.data.split import train_test_split
from recsys.evaluation.reporter import print_table
from recsys.evaluation.runner import evaluate_recommender
from recsys.recommenders import REGISTRY, build


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recommender", default="all", help=f"Recommender name or 'all'. Available: {list(REGISTRY)}")
    parser.add_argument("--k", type=int, default=TOP_K)
    parser.add_argument("--max-dialogs", type=int, default=None, help="Limit test dialogs (debug).")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (LLM-using recs only).")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing logs.")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    if args.recommender == "all":
        names = list(REGISTRY)
    else:
        names = [args.recommender]

    print("Loading dialogs...")
    dialogs = load_dialogs()
    train, test = train_test_split(dialogs)
    if args.max_dialogs:
        test = test[:args.max_dialogs]
    print(f"Train: {len(train)}  Test: {len(test)}")

    results = []
    for name in names:
        print(f"\n→ {name}")
        rec = build(name)
        workers = args.workers if rec.name.startswith(("llm_query_only", "hybrid")) else 1
        metrics = evaluate_recommender(
            rec, train, test,
            top_k=args.k, workers=workers, resume=not args.no_resume,
        )
        print(f"  HitRate@{args.k}: {metrics[f'HitRate@{args.k}']:.2f}%  "
              f"MRR: {metrics['MRR']:.2f}%  "
              f"NDCG@{args.k}: {metrics[f'NDCG@{args.k}']:.2f}%  "
              f"(n={metrics['n']})")
        results.append(metrics)

    if len(results) > 1:
        print_table(results, k=args.k)

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"eval_{ts}.json"
        out_path.write_text(json.dumps(results, indent=2))
        print(f"Results → {out_path}")


if __name__ == "__main__":
    main()
