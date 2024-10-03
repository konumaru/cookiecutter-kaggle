import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from config import Config
from utils import timer
from utils.feature import cache

FEATURE_DIR = Config().dirpath.feature

# =====================
# Target
# =====================


@cache(FEATURE_DIR)
def target() -> np.ndarray:
    data = pd.read_parquet(
        os.path.join(Config().dirpath.preprocessing, "data.parquet")
    )
    return data[[Config().target_name]].to_numpy()


# =====================
# Fold
# =====================
@cache(FEATURE_DIR)
def fold(cfg: Config) -> np.ndarray:
    X = pd.read_parquet(
        os.path.join(Config().dirpath.preprocessing, "data.parquet")
    )
    y = X[[Config().target_name]]

    result = np.zeros(len(y))
    cv = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    for i, (_, valid_idx) in enumerate(cv.split(X, y)):
        result[valid_idx] = i

    return result.reshape(-1, 1)


# =====================
# Feature
# =====================


@cache(FEATURE_DIR)
def basic_features() -> np.ndarray:
    data = pd.read_parquet(
        os.path.join(Config().dirpath.preprocessing, "data.parquet")
    )
    return data.drop(Config().target_name, axis=1).to_numpy()


def main() -> None:
    target()
    fold(Config())

    feat_funcs = [
        basic_features,
    ]
    for func in feat_funcs:
        func()


if __name__ == "__main__":
    with timer(os.path.basename(__file__)):
        main()
