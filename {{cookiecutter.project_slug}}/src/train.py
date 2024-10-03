import os
import pathlib
from typing import Optional

import joblib
import lightgbm
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from config import Config
from utils import timer
from utils.feature import load_feature
from utils.io import load_pickle, save_pickle, save_txt


def fit_xgb(
    params,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: Optional[np.ndarray],
    y_valid: Optional[np.ndarray],
    weight_train: Optional[np.ndarray] = None,
    weight_valid: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    save_filepath: Optional[str] = None,
) -> XGBRegressor:
    model = XGBRegressor(**params)
    model.set_params(random_state=seed)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=100,
        sample_weight=weight_train,
    )
    if save_filepath:
        model.save_model(save_filepath + ".json")
    return model


def fit_lgbm(
    params,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: Optional[np.ndarray],
    y_valid: Optional[np.ndarray],
    weight_train: Optional[np.ndarray] = None,
    weight_valid: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    save_filepath: Optional[str] = None,
) -> LGBMRegressor:
    model = LGBMRegressor(**params)
    model.set_params(random_state=seed)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[lightgbm.log_evaluation(50), lightgbm.early_stopping(50)],
        sample_weight=weight_train,
        eval_sample_weight=[weight_train, weight_valid],
    )
    if save_filepath:
        joblib.dump(model, save_filepath + ".joblib")
    return model


def fit_cat(
    params,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: Optional[np.ndarray],
    y_valid: Optional[np.ndarray],
    weight_train: Optional[np.ndarray] = None,
    weight_valid: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    save_filepath: Optional[str] = None,
) -> CatBoostRegressor:
    model = CatBoostRegressor(**params)
    model.set_params(random_state=seed)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=100,
        sample_weight=weight_train,
    )
    if save_filepath:
        model.save_model(save_filepath + ".cbm", format="cbm")
    return model


def train(cfg: Config) -> None:
    output_dir = pathlib.Path(cfg.dirpath.train)

    feature = load_feature(cfg.dirpath.feature, cfg.feature_names)
    target = load_pickle(
        f"{cfg.dirpath.feature}/{cfg.target_name}.pkl"
    ).ravel()
    fold = load_pickle(f"{cfg.dirpath.feature}/fold.pkl").ravel()
    oof = np.zeros(len(target))

    models = [
        ("xgb", cfg.model.xgb),
        ("lgbm", cfg.model.lgbm),
        ("cat", cfg.model.cat),
    ]

    for model_name, model_cfg in models:
        model_dir = output_dir / model_name / f"seed={cfg.seed}/"
        model_dir.mkdir(exist_ok=True, parents=True)

        for i in range(cfg.n_splits):
            print(f"Fold {i + 1}")

            train_idx = fold != i
            valid_idx = fold == i

            X_train = feature[train_idx]
            y_train = target[train_idx]
            X_valid = feature[valid_idx]
            y_valid = target[valid_idx]

            if model_name == "xgb":
                model = fit_xgb(
                    model_cfg.params,
                    X_train,
                    y_train,
                    X_valid,
                    y_valid,
                    seed=cfg.seed,
                    save_filepath=str(model_dir / str(i)),
                )
                oof[valid_idx] = model.predict(X_valid)

            elif model_name == "lgbm":
                model = fit_lgbm(
                    model_cfg.params,
                    X_train,
                    y_train,
                    X_valid,
                    y_valid,
                    seed=cfg.seed,
                    save_filepath=str(model_dir / str(i)),
                )
                oof[valid_idx] = model.predict(X_valid)

            elif model_name == "cat":
                model = fit_cat(
                    model_cfg.params,
                    X_train,
                    y_train,
                    X_valid,
                    y_valid,
                    seed=cfg.seed,
                    save_filepath=str(model_dir / str(i)),
                )
                oof[valid_idx] = model.predict(X_valid)

        save_pickle(oof, str(model_dir / "oof.pkl"))


def evaluate(cfg: Config) -> None:
    model_names = ["xgb", "lgbm", "cat"]

    for model_name in model_names:
        model_dir_suffix = f"{model_name}/seed={cfg.seed}/"
        model_dir = pathlib.Path(cfg.dirpath.train) / model_dir_suffix

        oof = load_pickle(str(model_dir / "oof.pkl"))
        target = load_pickle(f"{cfg.dirpath.feature}/{cfg.target_name}.pkl")

        score = mean_squared_error(target, oof, squared=False)
        print("Score:", score)
        save_txt(str(score), str(model_dir / f"score_{score:.8f}.txt"))


def main() -> None:
    config = Config()

    train(config)
    evaluate(config)


if __name__ == "__main__":
    with timer(os.path.basename(__file__)):
        main()
