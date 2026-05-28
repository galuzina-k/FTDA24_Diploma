import math


def hit_at_k(ranked: list[str], targets: list[str], k: int) -> float:
    return float(bool(set(ranked[:k]) & set(targets)))


def reciprocal_rank(ranked: list[str], targets: list[str]) -> float:
    target_set = set(targets)
    for i, rid in enumerate(ranked):
        if rid in target_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: list[str], targets: list[str], k: int) -> float:
    target_set = set(targets)
    dcg = sum(
        1.0 / math.log2(i + 2) for i, rid in enumerate(ranked[:k]) if rid in target_set
    )
    n_relevant = min(len(targets), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))
    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics(
    predictions: list[list[str]], targets: list[list[str]], k: int = 10
) -> dict:
    assert len(predictions) == len(targets)
    hits, rrs, ndcgs = [], [], []
    for pred, tgt in zip(predictions, targets):
        if not tgt:
            continue
        hits.append(hit_at_k(pred, tgt, k))
        rrs.append(reciprocal_rank(pred, tgt))
        ndcgs.append(ndcg_at_k(pred, tgt, k))
    n = len(hits)
    return {
        f"HitRate@{k}": 100 * sum(hits) / n if n else 0.0,
        "MRR": 100 * sum(rrs) / n if n else 0.0,
        f"NDCG@{k}": 100 * sum(ndcgs) / n if n else 0.0,
        "n": n,
    }
