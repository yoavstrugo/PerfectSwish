from pathlib import Path

import numpy as np


def get_data_dir() -> Path:
    """
    Returns the data directory for the project.
    :return:
    """
    return Path.home() / ".perfectswish"


def np_load(file_name):
    return np.load(get_data_dir() / file_name, allow_pickle=True)


def np_save(file_name, data):
    np.save(get_data_dir() / file_name, data)
