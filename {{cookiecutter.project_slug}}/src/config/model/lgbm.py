from typing import Any, Dict

from pydantic import BaseModel


class LgbmConfig(BaseModel):
    params: Dict[str, Any] = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.1,
        "num_leaves": 64,
        "subsample": 1,
        "colsample_bytree": 1,
        "reg_alpha": 0,
        "reg_lambda": 1,
    }
