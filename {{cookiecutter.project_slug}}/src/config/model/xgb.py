from typing import Any, Dict

from pydantic import BaseModel


class XgbConfig(BaseModel):
    params: Dict[str, Any] = {
        "device": "cuda",
        "booster": "gbtree",
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.1,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 1,
        "colsample_bytree": 1,
        "alpha": 0,
        "lambda": 1,
    }
