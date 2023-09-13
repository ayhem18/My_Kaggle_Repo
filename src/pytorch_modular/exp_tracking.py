"""
This scripts contains functionalities used to customize and automate the process of experiment tracking.
"""
import os
from typing import Union

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from src.pytorch_modular.directories_and_files import process_save_path, default_file_name
import pickle
import torch
import numpy as np
import random

HOME = os.getcwd()
RANDOM_SEED = 69


def set_seeds():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


def create_summary_writer(parent_dir: Union[str, Path],
                          experiment_name: str = None,
                          model_name: str = None,
                          return_path: bool = False) -> Union[SummaryWriter, tuple[SummaryWriter, Path]]:
    timestamp = default_file_name()
    # process the parent_dir first
    path = process_save_path(parent_dir, file_ok=False, dir_ok=True)

    if experiment_name is not None:
        path = os.path.join(path, experiment_name)

    if model_name is not None:
        path = os.path.join(path, model_name)

    os.makedirs(path, exist_ok=True)

    print(f"[INFO] Created SummaryWriter, saving to: {path}...")
    if return_path:
        return SummaryWriter(log_dir=path), Path(path)

    return SummaryWriter(log_dir=path)


def save_info(save_path: Union[Path, str],
              details: dict[str, object],
              details_folder: str = 'details'):
    save_path = process_save_path(os.path.join(save_path, details_folder), dir_ok=True, file_ok=False)

    for name, obj in details.items():
        p = os.path.join(save_path, (name + '.pkl'))
        with open(p, 'wb') as f:
            pickle.dump(obj, f)
