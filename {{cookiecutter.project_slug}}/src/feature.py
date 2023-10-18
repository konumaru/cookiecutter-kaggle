import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from utils import timer
from utils.feature import feature

FEATURE_DIR = "./data/feature"


@feature(FEATURE_DIR)
def dummy_target() -> np.ndarray:
    target = (np.random.rand(100) < 0.2).astype(int)
    return target.reshape(-1, 1)


@feature(FEATURE_DIR)
def dummy_feature() -> np.ndarray:
    data = np.zeros((100, 5))
    return data


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    dummy_target()

    feat_funcs = [
        dummy_feature,
    ]
    for func in feat_funcs:
        func()


if __name__ == "__main__":
    with timer("feature.py"):
        main()
