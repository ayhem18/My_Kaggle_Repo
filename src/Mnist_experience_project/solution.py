"""
This script is my main attempt to reach 100% accuracy on the Kaggle Mnist Dataset (test split of course xD)
"""
import os
import random

import pandas as pd
import numpy as np
import cv2 as cv
import torchvision.transforms as tr

from torch.optim import SGD, lr_scheduler

from typing import Union
from pathlib import Path

from src.Mnist_experience_project.model import BaselineModel
from src.pytorch_modular.image_classification import engine_classification as cls
from src.pytorch_modular import data_loaders as dl
from src.pytorch_modular import directories_and_files as dirf


def convert_csv_to_image(train_data_path: Union[Path, str],
                         test_data_path: Union[Path, str],
                         vis: bool = True):
    # first let's
    train_df = pd.read_csv(train_data_path, header=None)
    test_df = pd.read_csv(test_data_path, header=None)

    train_dir = os.path.join(Path(train_data_path).parent, 'train')
    test_dir = os.path.join(Path(test_data_path).parent, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    def train_row_to_image(row):
        label = row.iloc[0]
        data = np.array(row.iloc[1:], dtype=np.uint8).reshape((28, 28))
        label_dir = os.path.join(train_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        cv.imwrite(
            os.path.join(label_dir, f'image_{len(os.listdir(label_dir))}.jpg'), data)

        if vis:
            # the idea here is to display 5% of the generated images, well more or less
            p = random.random()
            if p <= 0.05:
                cv.imshow(f'{label}_image', data)
                cv.waitKey(0)
                cv.destroyAllWindows()

    def test_row_to_image(row):
        data = np.array(row, dtype=np.uint8).reshape((28, 28))
        cv.imwrite(
            os.path.join(test_dir, f'image_{len(os.listdir(test_dir))}.jpg'), data)

        if vis:
            # the idea here is to display 5% of the generated images, well more or less
            p = random.random()
            if p <= 0.05:
                cv.imshow(f'test_image', data)
                cv.waitKey(0)
                cv.destroyAllWindows()

    # apply the inner function to each row in the dataframe
    train_df.apply(train_row_to_image, axis=1)
    test_df.apply(test_row_to_image, axis=1)


def solution(convert: bool = False):
    train_df_path = os.path.join('data', 'mnist_train.csv')
    test_df_path = os.path.join('data', 'mnist_test.csv')

    train_dir = os.path.join(Path(train_df_path).parent, 'train')
    test_dir = os.path.join(Path(train_df_path).parent, 'val')

    if convert:
        convert_csv_to_image(train_df_path,
                             test_df_path)

    # initialize the model
    base_model = BaselineModel(input_shape=(28, 28), num_classes=10)

    baseline_preprocess = tr.Compose([
        tr.ToTensor(),
        # tr.Normalize(mean=[0.485, 0.456], std=[0.229, 0.224]),
    ])

    # create the dataloaders for the data
    train_dl, test_dl, _ = dl.create_dataloaders(train_dir=train_dir,
                                                 test_dir=test_dir,
                                                 train_transform=baseline_preprocess,
                                                 batch_size=64)

    # the train_model function requires at least 4 parameters
    optimizer = SGD(base_model.parameters(), momentum=0.99, lr=0.1)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.001, total_iters=200)
    train_configuration = {'optimizer': optimizer,
                           'scheduler': scheduler,
                           'min_val_loss': 10 ** -4,
                           'max_epochs': 100,
                           'report': True,
                           }

    results = cls.train_model(base_model, train_dl, test_dl, train_configuration, log_dir='runs',
                              save_path='saved_models')

    print(results)


if __name__ == '__main__':
    # train_df_path = os.path.join('data', 'mnist_train.csv')
    # test_df_path = os.path.join('data', 'mnist_test.csv')
    # convert_csv_to_image(train_df_path, test_df_path, vis=False)

    # print(len(os.listdir(os.path.join('data', 'train', '0'))))

    # validation_dir = dirf.dataset_portion(os.path.join('data', 'train'),
    #                                       os.path.join('data', 'val'),
    #                                       portion=0.1,
    #                                       copy=False)

    # time to train the model
    solution(convert=False)