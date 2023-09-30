import os
import sys
from pathlib import Path
from typing import Tuple
from torchvision import transforms

script_dir = os.path.dirname(os.path.realpath(__file__))
HOME = script_dir

DATA_FOLDER = os.path.join(script_dir, 'training', 'data')
current = script_dir
while 'src' not in os.listdir(current):
    current = Path(current).parent

sys.path.append(str(current))
sys.path.append(os.path.join(str(current), 'FaceSpoofing'))

import shutil
import src.pytorch_modular.directories_and_files as dirf


RAW_DATA = os.path.join(DATA_FOLDER, 'raw')


TRAIN_DIR = os.path.join(DATA_FOLDER, 'train')
TEST_DIR = os.path.join(DATA_FOLDER, 'test')


def set_up():
    if os.path.isdir(RAW_DATA):
        shutil.rmtree(RAW_DATA)

    dirf.unzip_data_file(os.path.join(DATA_FOLDER, 'raw.zip'))
    dirf.squeeze_directory(RAW_DATA)
    # remove any non directories in RAW_DATA folder
    # rename
    os.rename(os.path.join(RAW_DATA, 'ClientRaw'), os.path.join(RAW_DATA, 'real'))
    os.rename(os.path.join(RAW_DATA, 'ImposterRaw'), os.path.join(RAW_DATA, 'fake'))

    initial_raw_data = os.listdir(RAW_DATA)
    for f in initial_raw_data:
        pf = os.path.join(RAW_DATA, f)
        if not os.path.isdir(pf):
            
            os.remove(pf)

    raw1 = os.path.join(DATA_FOLDER, 'raw', 'real')
    raw0 = os.path.join(DATA_FOLDER, 'raw', 'fake')


    initial_dirs1 = os.listdir(raw1)
    initial_dirs0 = os.listdir(raw0)

    for raw1_dir in initial_dirs1:
        path_dir = os.path.join(raw1, raw1_dir)
        if os.path.isdir(path_dir) and not str(raw1_dir).endswith('.db'):
            dirf.copy_directories(path_dir, raw1, copy=False, filter_directories=lambda x: not x.endswith('.db'))   
        else:
            os.remove(path_dir)
        # remove the directory at the end
        shutil.rmtree(path_dir)
            
    for raw0_dir in initial_dirs0:
        path_dir = os.path.join(raw0, raw0_dir)

        if os.path.isdir(path_dir) and not str(raw0_dir).endswith('.db'):
            dirf.copy_directories(path_dir, raw0,copy=False, filter_directories=lambda x: not x.endswith('.db'))
        else:
            os.remove(path_dir)
        shutil.rmtree(path_dir)

    if os.path.isdir(TRAIN_DIR):
        shutil.rmtree(TRAIN_DIR)
        os.makedirs(TRAIN_DIR)

    if os.path.isdir(TEST_DIR):
        shutil.rmtree(TEST_DIR)
        os.makedirs(TEST_DIR)

    dirf.dataset_portion(RAW_DATA, TRAIN_DIR, portion=0.9, copy=False)
    dirf.copy_directories(RAW_DATA, TEST_DIR, copy=False)

    pr.prepare_data(TRAIN_DIR)
    pr.prepare_data(TEST_DIR)

# make a sanity check portion of the training dataset
dirf.dataset_portion(TRAIN_DIR, 
                    os.path.join(DATA_FOLDER, 'debug_portion'),
                    portion=0.05, 
                    copy=True)

from src.pytorch_modular.data_loaders import create_dataloaders
from FaceSpoofing.PixBiSupervision.training import preprocessing as pr, losses as ls
from  FaceSpoofing.PixBiSupervision.model import PiBiNet as binet,  DenseNetFE as dense
import src.pytorch_modular.image_classification.engine_classification as cls

from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import torch
from torch import nn


def output_layer(y_pred: Tuple[torch.Tensor, torch.Tensor]):
    _, logits = y_pred
    sigma = nn.Sigmoid()
    output = sigma(logits).to(torch.int32)
    return output.squeeze(dim=-1)


def compute_loss(y_pred: torch.Tensor, 
                 y: torch.Tensor,
                 alpha: float=0.25) -> torch.Tensor:
    bfm, logits = y_pred
    # feature_map_loss = ls.pixelBinaryLoss()(bfm, y.float())
    binary_loss = nn.BCEWithLogitsLoss()(logits.squeeze(), y.float())
    return binary_loss
    # return alpha * feature_map_loss + binary_loss * (1 - alpha)


def main(set_data: bool = False,
        train_dir=TRAIN_DIR, 
        test_dir=TEST_DIR):
    if set_data:
        set_up()

    preprocess = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # prepare the data loaders
    train_dl, test_dl, _ = create_dataloaders(train_dir=train_dir, 
                                                test_dir=test_dir,
                                                train_transform=preprocess,
                                                batch_size=128)
    
    model = binet.PiBiNet()
    optimizer = AdamW(params=model.parameters(), lr=0.01, weight_decay=10**-5)
    lr_scheduler = LinearLR(optimizer=optimizer, start_factor=1, end_factor=0.5, total_iters=100)

    # accuracy_metric = Accuracy(task='binary')
    # metrics = {'accuracy': accuracy_metric}

    train_configuration = {'optimizer': optimizer,
                            'scheduler': lr_scheduler,
                            'min_val_loss': 10 ** -4,
                            'max_epochs': 10,
                            'report_epoch': 1,
                            'compute_loss': compute_loss,
                            'no_improve_stop': 10,
                            'output_layer': output_layer, 
                            'debug': False,
                            }

    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    results = cls.train_model(model, 
                            train_dl, 
                            test_dl, 
                            train_configuration,          
                            log_dir=os.path.join(HOME, 'runs'),         
                            save_path=os.path.join(HOME, 'saved_models'))   


if __name__ == '__main__':
    check_pnt = os.path.join(DATA_FOLDER, 'debug_portion')
    main(set_data=False, 
        train_dir=check_pnt, 
        test_dir=TEST_DIR)
    