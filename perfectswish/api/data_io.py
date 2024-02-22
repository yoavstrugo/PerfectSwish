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


def load_data(file_name) -> np.ndarray:
    return np.load(get_data_dir() / file_name, allow_pickle=True)


def save_data(file_name, data) -> None:
    np.save(get_data_dir() / file_name, data)
