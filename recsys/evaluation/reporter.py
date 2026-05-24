from recsys.config import TOP_K


def print_table(results: list[dict], k: int = TOP_K) -> None:
    hr_col = f"HitRate@{k}"
    ndcg_col = f"NDCG@{k}"

    header = f"{'Recommender':<22} {hr_col:>12} {'MRR':>10} {ndcg_col:>12} {'n':>6}"
    print()
    print(header)
    print("-" * len(header))

    for r in sorted(results, key=lambda x: x.get(ndcg_col, 0.0), reverse=True):
        print(
            f"{r['recommender']:<22} "
            f"{r.get(hr_col, 0.0):>11.2f}% "
            f"{r.get('MRR', 0.0):>9.2f}% "
            f"{r.get(ndcg_col, 0.0):>11.2f}% "
            f"{r.get('n', 0):>6}"
        )
    print()
