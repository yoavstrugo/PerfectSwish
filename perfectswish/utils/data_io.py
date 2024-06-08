import pickle
from pathlib import Path

import numpy as np

__DATA_DIR = Path.home() / ".perfectswish"


def get_data_dir() -> Path:
    """
    Returns the data directory for the project.
    :return:
    """
    data_dir = __DATA_DIR

    if not data_dir.exists():
        data_dir.mkdir()

    return data_dir


def load_data(file_name):
    # load file_name with pickle
    with open(get_data_dir() / file_name, "rb") as f:
        data = pickle.load(f)

    return data


def save_data(file_name, data):
    # save data to file_name with pickle
    with open(get_data_dir() / file_name, "wb") as f:
        pickle.dump(data, f)
