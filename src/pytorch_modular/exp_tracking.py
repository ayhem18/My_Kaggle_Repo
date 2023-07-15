"""
This scripts contains functionalities used to customize and automate the process of experiment tracking.
"""
import os
from typing import Union
from datetime import datetime 
HOME = os.getcwd()
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from src.pytorch_modular.directories_and_files import process_save_path, default_file_name
import pickle
import torch
import numpy as np
import random 

RANDOM_SEED = 69

def set_seeds():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    


def create_summary_writer(parent_dir: Union[str, Path],
                         experiment_name: str,
                         model_name: str, 
                         return_path:bool=False) -> Union[SummaryWriter, tuple[SummaryWriter, Path]]:
    timestamp = default_file_name()
    # build the path for the log directory
    path = os.path.join(parent_dir, experiment_name, model_name, timestamp)
    # process the path
    log_dir = process_save_path(path, file_ok=False, dir_ok=True)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    if return_path:
        return SummaryWriter(log_dir=log_dir), Path(log_dir)

    return SummaryWriter(log_dir=log_dir)


def save_info(save_path: Union[Path, str],
              details: dict[str, object],
              details_folder: str='details'):
    save_path = process_save_path(os.path.join(save_path, details_folder), dir_ok=True, file_ok=False)    

    for name, obj in details.items():
        p = os.path.join(save_path, (name + '.p'))
        with open(p, 'wb') as f:
            pickle.dump(obj, f)
