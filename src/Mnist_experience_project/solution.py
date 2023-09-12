"""
This script is my main attempt to reach 100% accuracy on the Kaggle Mnist Dataset (test split of course xD)
"""
import os
import random
import sys

import pandas as pd
import numpy as np
import cv2 as cv
import torchvision.transforms as tr

from torch.optim import SGD, lr_scheduler

from pathlib import Path
from torch import nn

# before proceeding with the src. inputs, I need to add it to the PATH environment variable
# for the interpreter to find it

try:
    from src.pytorch_modular.pytorch_utilities import load_model
    from src.Mnist_experience_project.model import BaselineModel
    from src.pytorch_modular.image_classification import engine_classification as cls
    from src.pytorch_modular import data_loaders as dl
    from src.pytorch_modular import directories_and_files as dirf
    from src.pytorch_modular.image_classification import engine_classification as en_c

except ModuleNotFoundError:
    # the idea here is simple, climb in the file system hierarchy until the 'src' folder is detected
    current = Path(os.getcwd())
    while 'src' not in os.listdir(current):
        current = current.parent

    # now add the 'src' folder to the PATH variable
    sys.path.append(str(current))
    sys.path.append(str(os.path.join(current, 'src')))

    from src.pytorch_modular.pytorch_utilities import load_model
    from src.Mnist_experience_project.model import BaselineModel
    from src.pytorch_modular.image_classification import engine_classification as cls
    from src.pytorch_modular import data_loaders as dl
    from src.pytorch_modular import directories_and_files as dirf
    from src.pytorch_modular.image_classification import engine_classification as en_c


def convert_csv_to_image(vis: bool = False):
    data_folder = '/home/ayhem18/DEV/My_Kaggle_Repo/src/Mnist_experience_project/data'
    train_data_path = os.path.join(data_folder, 'mnist_train.csv')
    test_data_path = os.path.join(data_folder, 'mnist_test.csv')

    # first let's
    train_df = pd.read_csv(train_data_path, header=None)
    test_df = pd.read_csv(test_data_path, header=None)

    train_dir = os.path.join(Path(train_data_path).parent, 'train')
    test_dir = os.path.join(Path(test_data_path).parent, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    def train_row_to_image(row):
        label = row.iloc[0]
        data = np.array(row.iloc[1:], dtype=np.float32).reshape((28, 28)) / 255  # normalize the dataset
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
        data = np.array(row, dtype=np.float32).reshape((28, 28)) / 255  # normalize the dataset
        cv.imwrite(
            os.path.join(test_dir, f'image_{len(os.listdir(test_dir))}.jpg'), data)

        if vis:
            # the idea here is to display 0.1% of the generated images, well more or less
            p = random.random()
            if p <= 0.001:
                cv.imshow(f'test_image', data)
                cv.waitKey(0)
                cv.destroyAllWindows()

    # apply the inner function to each row in the dataframe
    train_df.apply(train_row_to_image, axis=1)
    test_df.apply(test_row_to_image, axis=1)

    # leave a portion of the training split as a validation split
    # create the training
    val_dir = os.path.join(data_folder, 'val')
    os.makedirs(val_dir, exist_ok=True)
    dirf.dataset_portion(train_dir,
                         destination_directory=val_dir,
                         portion=0.15, copy=False)


def solution():
    data_folder = '/home/ayhem18/DEV/My_Kaggle_Repo/src/Mnist_experience_project/data'
    train_dir = os.path.join(data_folder, 'train')
    val_dir = os.path.join(data_folder, 'val')

    # initialize the model
    base_model = BaselineModel(input_shape=(28, 28),
                               num_classes=10,
                               num_conv_blocks=0)

    baseline_preprocess = tr.Compose([
        tr.ToTensor(),
        # tr.Normalize(mean=[0.255], std=[0.229, 0.224]),
    ])

    # create the dataloaders for the data
    train_dl, test_dl, _ = dl.create_dataloaders(train_dir=train_dir,
                                                 test_dir=val_dir,
                                                 train_transform=baseline_preprocess,
                                                 batch_size=128)

    # the train_model function requires at least 4 parameters
    optimizer = SGD(base_model.parameters(), momentum=0.99, lr=0.05)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=100)
    train_configuration = {'optimizer': optimizer,
                           'scheduler': scheduler,
                           'min_val_loss': 10 ** -4,
                           'max_epochs': 50,
                           'report_epoch': 5,
                           }

    script_dir = os.path.dirname(os.path.realpath(__file__))

    results = cls.train_model(base_model, train_dl, test_dl, train_configuration,
                              log_dir=os.path.join(script_dir, 'runs'),
                              save_path=os.path.join(script_dir, 'saved_models'))

    print(results['val_loss'][-5:])
    print(results['val_accuracy'][-5:])

    test_dir = os.path.join(script_dir, 'data', 'test')

    # run the inference
    predictions = cls.inference(base_model,
                                test_dir,
                                baseline_preprocess,
                                return_tensor='list')

    # save the predictions to a file
    submission = pd.DataFrame(data={'id': list(range(len(predictions))),
                                    'label': predictions})

    sub_folder = os.path.join(script_dir, 'submissions')
    submission.to_csv(os.path.join(sub_folder, f'sub_{len(os.listdir(sub_folder))}.csv'), index=False)


def verify_performance(model):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_folder = os.path.join(script_dir, 'data')

    baseline_preprocess = tr.Compose([
        # the images are gray scale
        tr.Grayscale(num_output_channels=1),
        tr.ToTensor(),
        # tr.Normalize(mean=[0.255], std=[0.229, 0.224]),
    ])

    val_dir = os.path.join(data_folder, 'val')

    val_loader, _ = dl.create_dataloaders(train_dir=val_dir,
                                          train_transform=baseline_preprocess,
                                          batch_size=500)

    def output_layer(x):
        return x.argmax(dim=-1)

    en_c.val_per_epoch(model=model,
                       dataloader=val_loader,
                       loss_function=nn.CrossEntropyLoss(),
                       output_layer=output_layer,
                       debug=True)


if __name__ == '__main__':
    # convert_csv_to_image(vis=False)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_folder = os.path.join(script_dir, 'data')
    base_model = BaselineModel(input_shape=(28, 28),
                               num_classes=10,
                               num_conv_blocks=0)
    model = load_model(base_model=base_model,
                       path=os.path.join(script_dir, 'saved_models', '9-9-22-58.pt'))
    