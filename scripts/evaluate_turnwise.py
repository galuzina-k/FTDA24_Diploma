"""Turn-level evaluation.

For each recommender, evaluate at multiple dialog-history cutoffs (measured in
seeker turns: 2, 4, 6, 8, full). At each cutoff, dialogs are truncated to the
Nth seeker turn — so we measure "given only the first N seeker turns, can the
recommender already predict the eventual recommendation?"

The annotated `user_query` was generated from the FULL dialog and would leak
future information at intermediate cutoffs, so we pass query="" at every
intermediate cutoff and only pass the real annotated query at "full".

Logs are written per (recommender, cutoff) so the run is resume-safe and the
data can be re-analyzed without re-running.
"""

import argparse
import copy
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recsys.config import TOP_K
from recsys.data.inspired import Dialog, Turn
from recsys.data.split import train_test_split
from recsys.data.inspired import load_dialogs, dialog_context_before_recommendation
from recsys.evaluation.runner import evaluate_recommender
from recsys.recommenders import REGISTRY, build

DEFAULT_RECOMMENDERS = [
    "popularity",
    "svd",
    "semantic_retrieval",
    "history_only",
    "llm_query_only",
]
DEFAULT_CUTOFFS = [2, 4, 6, 8, "full"]


def _truncate_to_seeker_turns(dialog: Dialog, n_seeker: int | None) -> Dialog | None:
    """Return a copy of `dialog` whose turns are cut after the Nth seeker turn
    (inclusive). If n_seeker is None, use the full pre-recommendation context.

    Returns None if the dialog has fewer than `n_seeker` seeker turns before
    the recommendation.
    """
    pre_rec = dialog_context_before_recommendation(dialog)
    if n_seeker is None:
        truncated = pre_rec
    else:
        seeker_seen = 0
        cutoff_idx = None
        for i, t in enumerate(pre_rec):
            if t.role == "seeker" and t.text.strip():
                seeker_seen += 1
                if seeker_seen == n_seeker:
                    cutoff_idx = i + 1
                    break
        if cutoff_idx is None:
            return None
        truncated = pre_rec[:cutoff_idx]

    # Shallow-copy the dialog with truncated turns; clear user_query for
    # intermediate cutoffs (it was generated from the full dialog).
    new_dialog = copy.copy(dialog)
    new_dialog.turns = list(truncated)
    if n_seeker is not None:
        new_dialog.user_query = ""
        # Filter history_imdb_ids to only movies mentioned in the truncated
        # turns. Without this, recommenders that use history_imdb_ids (SVD,
        # HistoryOnly) receive full-dialog movie context at every cutoff,
        # making their cutoff curves meaningless.
        combined_text = " ".join(t.text for t in truncated).lower()
        seen: set[str] = set()
        filtered: list[str] = []
        for title, imdb_id in new_dialog.mention_map.items():
            if title.lower() in combined_text and imdb_id not in seen:
                filtered.append(imdb_id)
                seen.add(imdb_id)
        new_dialog.history_imdb_ids = filtered
    return new_dialog


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recommenders",
        nargs="+",
        default=DEFAULT_RECOMMENDERS,
        help=f"Recommender names. Available: {list(REGISTRY)}",
    )
    parser.add_argument(
        "--cutoffs",
        nargs="+",
        type=int,
        default=DEFAULT_CUTOFFS,
        help="Seeker-turn cutoffs",
    )
    parser.add_argument("--k", type=int, default=TOP_K)
    parser.add_argument("--max-dialogs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    print("Loading dialogs...")
    dialogs = load_dialogs()
    train, test = train_test_split(dialogs)
    if args.max_dialogs:
        test = test[: args.max_dialogs]
    print(f"Train: {len(train)}  Test: {len(test)}")

    cutoffs: list[int | None] = list(args.cutoffs)

    # results[recommender][cutoff_label] = metrics dict
    results: dict[str, dict[str, dict]] = defaultdict(dict)

    for cutoff in cutoffs:
        label = "full" if cutoff == "full" else f"t{cutoff}"
        truncated_test = []
        for d in test:
            t = _truncate_to_seeker_turns(d, cutoff)
            if t is not None:
                truncated_test.append(t)

        for name in args.recommenders:
            print(f"\n→ {name} @ cutoff={label}  (n_dialogs={len(truncated_test)})")
            rec = build(name)
            original_name = rec.name
            rec.name = f"{original_name}__cutoff_{label}"

            workers = (
                args.workers
                if original_name.startswith(("llm_query_only", "hybrid"))
                else 1
            )
            metrics = evaluate_recommender(
                rec,
                train,
                truncated_test,
                top_k=args.k,
                workers=workers,
                resume=not args.no_resume,
            )
            print(
                f"  HR@{args.k}: {metrics[f'HitRate@{args.k}']:.2f}%  "
                f"MRR: {metrics['MRR']:.2f}%  "
                f"NDCG@{args.k}: {metrics[f'NDCG@{args.k}']:.2f}%  "
                f"(n={metrics['n']})"
            )
            results[original_name][label] = metrics

    _print_table(results, args.k, cutoffs)


def _print_table(results: dict, k: int, cutoffs: list):
    labels = ["full" if c is None else f"t{c}" for c in cutoffs]
    print("\n" + "=" * 78)
    print(f"Turn-level evaluation summary (metric @ k={k})")
    print("=" * 78)
    for metric_key in (f"HitRate@{k}", "MRR", f"NDCG@{k}"):
        print(f"\n{metric_key}")
        header = (
            f"  {'recommender':<24}"
            + "".join(f"  {lab:>8}" for lab in labels)
            + f"  {'(n_full)':>10}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, by_cutoff in results.items():
            row = f"  {name:<24}"
            for lab in labels:
                m = by_cutoff.get(lab)
                row += f"  {m[metric_key]:>7.2f}%" if m else f"  {'-':>8}"
            n_full = by_cutoff.get("full", {}).get("n", "-")
            row += f"  {str(n_full):>10}"
            print(row)


if __name__ == "__main__":
    main()
