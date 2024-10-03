import os
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from .model.cat import CatConfig
from .model.lgbm import LgbmConfig
from .model.xgb import XgbConfig


class DirPath(BaseModel):
    root: str = "data"
    raw: str = os.path.join(root, "raw")
    preprocessing: str = os.path.join(root, "preprocessing")
    feature: str = os.path.join(root, "feature")
    train: str = os.path.join(root, "train")
    ensemble: str = os.path.join(root, "ensemble")
    external: str = os.path.join(root, "external")


class Model(BaseModel):
    xgb: XgbConfig = XgbConfig()
    lgbm: LgbmConfig = LgbmConfig()
    cat: CatConfig = CatConfig()


class Config(BaseModel):
    seed: int = 42
    n_splits: int = 5
    target_name: str = "target"

    dirpath: DirPath = DirPath()

    feature_names: List[str] = Field(
        default_factory=lambda: [
            "basic_features",
        ]
    )

    model: Model = Model()
