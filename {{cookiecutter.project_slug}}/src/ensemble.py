import os
import pathlib
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from config import Config
from utils import timer
from utils.io import load_pickle, save_pickle


def fit_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: Optional[int] = None,
    save_filepath: Optional[str] = None,
) -> Ridge:
    model = Ridge(
        alpha=10.0,
        fit_intercept=False,
        random_state=seed,
    )
    model.fit(X_train, y_train.ravel())
    print("Ridge Coef:", model.coef_)

    if save_filepath:
        joblib.dump(model, save_filepath + ".joblib")

    return model


def load_oof(cfg: Config) -> pd.DataFrame:
    model_names = ["xgb", "lgbm", "cat"]

    oof = {}
    for model_name in model_names:
        model_dir_suffix = f"{model_name}/seed={cfg.seed}/"
        model_dir = pathlib.Path(cfg.dirpath.train) / model_dir_suffix

        oof[model_name] = load_pickle(str(model_dir / "oof.pkl"))

    return pd.DataFrame(oof)


def train(cfg: Config) -> np.ndarray:
    oof = load_oof(cfg)
    target = load_pickle(
        f"{cfg.dirpath.feature}/{cfg.target_name}.pkl"
    ).ravel()
    fold = load_pickle(f"{cfg.dirpath.feature}/fold.pkl").ravel()
    oof_ensemble = np.zeros(len(target))

    for i in range(cfg.n_splits):
        print(f"Fold {i + 1}")

        train_idx = fold != i
        valid_idx = fold == i

        X_train = oof[train_idx]
        y_train = target[train_idx]
        X_valid = oof[valid_idx]

        model = fit_ridge(X_train, y_train, seed=cfg.seed)
        oof_ensemble[valid_idx] = model.predict(X_valid)

    output_dir = pathlib.Path(cfg.dirpath.ensemble)
    output_dir.mkdir(exist_ok=True, parents=True)
    save_pickle(oof_ensemble, output_dir / "oof.pkl")
    return oof_ensemble


def evaluate(cfg: Config) -> None:
    oof_en = load_pickle(f"{cfg.dirpath.ensemble}/oof.pkl")
    target = load_pickle(
        f"{cfg.dirpath.feature}/{cfg.target_name}.pkl"
    ).ravel()

    score = mean_squared_error(target, oof_en, squared=False)
    print(f"RMSE: {score:.6f}")


def main() -> None:
    config = Config()

    train(config)
    evaluate(config)


if __name__ == "__main__":
    with timer(os.path.basename(__file__)):
        main()
