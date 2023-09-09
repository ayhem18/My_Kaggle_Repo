"""
This script contains a number of functionalities to debug the training of image classifier
"""
import torch

from pathlib import Path
from typing import Union

from src.pytorch_modular.directories_and_files import process_save_path

def save_random_train_predictions(directory: Union[Path, str],
                                  samples: torch.Tensor, 
                                  labels: torch.Tensor,
                                  ) -> None:
    # first let's set the directory
    directory = process_save_path(directory, file_ok=False, dir_ok=True)    

    # this function will be run only 1% of the time (on average)
    if random.random() <= 0.01:


