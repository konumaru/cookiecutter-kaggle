import pathlib
from typing import Union

import hydra
import lightgbm
import numpy as np
from lightgbm import LGBMRegressor
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from utils import timer
from utils.feature import load_feature
from utils.io import load_pickle, save_pickle, save_txt


def fit_rf(
    params,
    save_filepath: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    seed: int = 42,
) -> RandomForestRegressor:
    model = RandomForestRegressor(**params)
    model.set_params(random_state=seed)
    model.fit(X_train, y_train)
    save_pickle(str(f"{save_filepath}.pkl"), model)
    return model


def fit_xgb(
    params,
    save_filepath: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: Union[np.ndarray, None] = None,
    y_valid: Union[np.ndarray, None] = None,
    seed: int = 42,
) -> XGBRegressor:
    model = XGBRegressor(**params)
    model.set_params(random_state=seed)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=50,
    )
    model.save_model(save_filepath + ".json")
    return model


def fit_lgbm(
    params,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    save_filepath: str,
    seed: int = 42,
) -> LGBMRegressor:
    model = LGBMRegressor(**params)
    model.set_params(random_state=seed)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[lightgbm.log_evaluation(50), lightgbm.early_stopping(50)],
    )
    model.booster_.save_model(  # type: ignore
        save_filepath + ".txt",
        num_iteration=model.best_iteration_,
        importance_type="gain",
    )
    return model


def train(cfg: DictConfig) -> None:
    model_dir_suffix = f"{cfg.model.name}/seed={cfg.seed}/"
    model_dir = pathlib.Path(cfg.path.model) / model_dir_suffix
    model_dir.mkdir(exist_ok=True, parents=True)

    feature = load_feature(cfg.path.feature, cfg.feature_names)
    target = load_pickle(f"{cfg.path.feature}/{cfg.target_name}.pkl")

    oof = np.zeros(len(feature))

    cv = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    for i, (train_idx, valid_idx) in enumerate(cv.split(feature)):
        print(f"Fold {i + 1}")

        X_train = feature[train_idx]
        y_train = target[train_idx]
        X_valid = feature[valid_idx]
        y_valid = target[valid_idx]

        if cfg.model.name == "xgb":
            model = fit_xgb(
                cfg.model.params,
                str(model_dir / str(i)),
                X_train,
                y_train,
                X_valid,
                y_valid,
                cfg.seed,
            )
            oof[valid_idx] = model.predict(X_valid)

        elif cfg.model.name == "rf":
            model = fit_rf(
                cfg.model.params,
                str(model_dir / str(i)),
                X_train,
                y_train,
                X_valid,
                y_valid,
                cfg.seed,
            )
            oof[valid_idx] = model.predict(X_valid)

    save_pickle(str(model_dir / "oof.pkl"), oof)


def evaluate(cfg: DictConfig) -> None:
    model_dir_suffix = f"{cfg.model.name}/seed={cfg.seed}/"
    model_dir = pathlib.Path(cfg.path.model) / model_dir_suffix

    oof = load_pickle(str(model_dir / "oof.pkl"))
    target = load_pickle(f"{cfg.path.feature}/{cfg.target_name}.pkl")

    print(oof.shape)

    score = mean_squared_error(target, oof, squared=False)
    print(score)
    save_txt(
        str(model_dir / f"score_{score:.8f}.txt"),
        str(score),
    )


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    train(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    with timer("train.py"):
        main()
