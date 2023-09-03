import pathlib
from typing import Any, Union

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestRegressor

from utils import timer
from utils.io import load_pickle, save_pickle


def get_model(model_name: str = "rf", seed: int = 42) -> Any:
    return RandomForestRegressor(n_estimators=100, random_state=seed)


def train(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: Union[np.ndarray, None] = None,
    y_valid: Union[np.ndarray, None] = None,
    gruop_train: Union[np.ndarray, None] = None,
    gruop_valid: Union[np.ndarray, None] = None,
    seed=42,
):
    model_dir = pathlib.Path(f"./data/model/{model_name}/seed={seed}")
    model_dir.mkdir(exist_ok=True)

    model = get_model(model_name, seed)
    model.fit(X_train, y_train)
    save_pickle(str(model_dir / "model.pkl"), model)


def predict(
    model_name: str,
    X_valid: Union[np.ndarray, None] = None,
    y_valid: Union[np.ndarray, None] = None,
    gruop_valid: Union[np.ndarray, None] = None,
    seed: int = 42,
) -> np.ndarray:
    model_dir = pathlib.Path(f"./data/model/{model_name}/seed={seed}")

    pred = np.zeros_like(y_valid)
    model = load_pickle(str(model_dir / "model.pkl"))
    pred = model.predict(X_valid)

    return pred


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print("Hello Workd!")


if __name__ == "__main__":
    with timer("main.py"):
        main()
