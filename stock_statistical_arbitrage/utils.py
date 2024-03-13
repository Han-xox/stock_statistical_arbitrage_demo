import importlib
from copy import deepcopy
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml


def annualized_return(r: float, T: int, DoY: int = 255):
    R = (1 + r) ** (DoY / T) - 1
    return R


def annualized_sharp_ratio(s: float, T: int, DoY=255):
    S = s * np.sqrt(DoY / T)
    return S


def read_from_yaml(filename: str) -> dict:
    """read yaml file and return a dict"""

    filename = Path(filename)

    if filename.suffix not in [".yaml", ".yml"]:
        raise FileNotFoundError(f"File [{filename}] is not yaml file.")

    with filename.open(mode="r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise RuntimeError(f"Reading yaml failed: {exc}")


def init_obj_dict(obj_dict: dict):
    # don't touch the original dict
    obj_dict = deepcopy(obj_dict)

    # Extract the class path and init args from the dictionary
    class_path = obj_dict.get("class_path", None)
    init_args = obj_dict.get("init_args", None)

    if class_path is None or init_args is None:
        return obj_dict

    # Split the class path into module and class name
    module_name, class_name = class_path.rsplit(".", 1)

    # Import the module dynamically
    module = importlib.import_module(module_name)

    # Get the class from the module
    cls = getattr(module, class_name)

    # Recursively create nested objects
    for key, value in init_args.items():
        if isinstance(value, dict):
            init_args[key] = init_obj_dict(value)

    # Create an instance of the class using the init args
    obj = cls(**init_args)

    return obj


class CategoryEncoder:
    """Encode categorical variables to int with an ascending manner"""

    def __init__(self) -> None:
        self._cat: Dict[str, Dict[str, int]] = {}

    def fit_transform(self, df: pd.DataFrame):
        for c in df.columns:
            _map = self._cat.get(c, None)
            if _map:
                counter = len(_map)
                _map.update(
                    {
                        k: i + counter
                        for i, k in enumerate(
                            sorted(set(df[c].unique()) - set(_map.keys()))
                        )
                    }
                )
            else:
                self._cat[c] = {k: i for i, k in enumerate(sorted(df[c].unique()))}

        return pd.DataFrame(data={c: df[c].map(self._cat[c]) for c in df.columns})
