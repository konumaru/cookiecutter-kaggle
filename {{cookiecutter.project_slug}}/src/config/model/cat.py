from typing import Any, Dict

from pydantic import BaseModel


class CatConfig(BaseModel):
    params: Dict[str, Any] = {
        "task_type": "CPU",
        "loss_function": "RMSE",
        # "learning_rate": 0.08,
        # "iterations": 2000,
        # "bagging_temperature": 0.5,
        # "max_depth": 12,
        # "l2_leaf_reg": 1.25,
        # "min_data_in_leaf": 24,
        # "random_strength": 0.25,
        # "use_best_model": True,
        "iterations": 100,
        "learning_rate": 0.026093623304651172,
        "bagging_temperature": 0.32105624997614,
        "max_depth": 11,
        "l2_leaf_reg": 1.207224144365719,
        "min_data_in_leaf": 33,
    }
