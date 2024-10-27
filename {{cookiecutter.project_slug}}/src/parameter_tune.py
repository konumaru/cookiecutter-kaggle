import os

import numpy as np
import optuna
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from config import Config
from utils import timer
from utils.feature import load_feature
from utils.io import load_pickle


def objective_lgbm(trial: optuna.Trial, cfg: Config) -> float:
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": trial.suggest_categorical("metric", ["rmse", "mse"]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "n_estimators": 2000,
        "max_depth": trial.suggest_int("max_depth", 1, 12),
        "num_leaves": trial.suggest_int("num_leaves", 32, 128),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0),
        "force_col_wise": True,
        "seed": 42,
    }

    feature = load_feature(
        "data/feature",
        sorted(["agent_parsed_feature", "numeric_feature"]),
    )

    feature = load_feature(cfg.dirpath.feature, cfg.feature_names)
    target = load_pickle(
        f"{cfg.dirpath.feature}/{cfg.target_name}.pkl"
    ).ravel()

    oof = np.zeros(len(target))
    cv = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    for i, (train_idx, valid_idx) in enumerate(cv.split(feature, target)):
        X_train = feature[train_idx]
        y_train = target[train_idx]
        X_valid = feature[valid_idx]
        y_valid = target[valid_idx]

        model = LGBMRegressor(**params, silent=True, verbose=-1)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
        )
        oof[valid_idx] = model.predict(X_valid)

    rmse = mean_squared_error(target.to_numpy(), oof, squared=False)  # type: ignore

    return float(rmse)


def main() -> None:
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective_lgbm(trial, Config()), n_trials=100)
    print("Best hyperparameters:", study.best_params)
    print("Best RMSE:", study.best_value)


if __name__ == "__main__":
    with timer(os.path.basename(__file__)):
        main()
