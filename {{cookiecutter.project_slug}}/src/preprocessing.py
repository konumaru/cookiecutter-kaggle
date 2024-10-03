import os

import pandas as pd
from sklearn.datasets import load_diabetes

from config import Config
from utils import timer


def main() -> None:
    config = Config()
    print(config.model_dump_json(indent=4))

    X, y = load_diabetes(return_X_y=True, as_frame=True)

    raw_data: pd.DataFrame = X.copy()  # type: ignore
    raw_data[config.target_name] = y

    print(raw_data.head())

    raw_data.to_parquet(
        os.path.join(config.dirpath.preprocessing, "data.parquet")
    )


if __name__ == "__main__":
    with timer(os.path.basename(__file__)):
        main()
