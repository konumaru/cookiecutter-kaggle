import os

import hydra
from omegaconf import DictConfig, OmegaConf

from utils import timer


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print("Hello Workd!")


if __name__ == "__main__":
    with timer(os.path.basename(__file__)):
        main()
