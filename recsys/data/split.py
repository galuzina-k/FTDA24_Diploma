import random

from recsys.config import SPLIT_SEED, SPLIT_TEST_RATIO
from recsys.data.inspired import Dialog


def train_test_split(
    dialogs: list[Dialog],
    test_ratio: float = SPLIT_TEST_RATIO,
    seed: int = SPLIT_SEED,
) -> tuple[list[Dialog], list[Dialog]]:
    data = list(dialogs)
    random.Random(seed).shuffle(data)
    split = int(len(data) * (1 - test_ratio))
    return data[:split], data[split:]
