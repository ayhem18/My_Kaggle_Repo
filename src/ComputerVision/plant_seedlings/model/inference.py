import os, sys

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Union
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from src.pytorch_modular.directories_and_files import process_save_path
from src.pytorch_modular.image_classification.engine_classification import inference, binary_output


class InferenceDataset(Dataset):
    def __init__(self, test_dir: Union[str, Path], transformations: T) -> None:
        # the usual base class constructor call
        super().__init__()
        test_data_path = process_save_path(test_dir, file_ok=False, dir_ok=True)
        data = [os.path.join(test_data_path, file_name) for file_name in os.listdir(test_data_path)]
        self.data = sorted(data, key=lambda p: int(os.path.basename(p)[:-4]))
        self.t = transformations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> int:
        # don't forget to apply the transformation before returning the index-th element in the directory
        pil_image = Image.open(self.data[index])
        item = self.t(pil_image)
        return item


def inference(classifier: dvc.DVC_Classifier,
              test_dir: Union[Path, str]) -> Union[np.ndarray, torch.Tensor, list[int]]:
    # first initialize a dataset with the test_dir
    ds = InferenceDataset(test_dir, classifier.get_transformations())

    # we need a dataloader for batched inference
    dataloader = DataLoader(ds, batch_size=100, shuffle=False,
                            num_workers=os.cpu_count() // 2)  # setting shuffle to False, cause the objective of this dataloader is not training

    # get the predictions
    predictions = inference(classifier, dataloader, lambda x: binary_output(x))

    return predictions


def submission(predictions: np.ndarray, save_path: Union[Path, str]):
    # create a pandas dataframe to format the predictions
    sub = pd.DataFrame({"id": list(range(1, len(predictions) + 1)), 'labels': list(predictions)})

    save_path = process_save_path(save_path,
                                  dir_ok=True,
                                  file_ok=True,
                                  condition=lambda p: not os.path.isfile(p) or os.path.basename(p).endswith('.csv'))

    if os.path.isdir(save_path):
        save_path = os.path.join(save_path, 'submission.csv')

    sub.to_csv(save_path, index=False)
    return sub

