import pathlib
import pickle
from typing import Any, Union


def save_pickle(data: Any, filepath: Union[str, pathlib.Path]) -> None:
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_pickle(filepath: Union[str, pathlib.Path]) -> Any:
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def save_txt(data: Any, filepath: Union[str, pathlib.Path]) -> None:
    with open(filepath, "w") as f:
        f.write(data)


def load_txt(filepath: Union[str, pathlib.Path]) -> Any:
    with open(filepath, "r") as f:
        data = f.read()
    return data
