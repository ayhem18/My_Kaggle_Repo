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


RAW_DATA = os.path.join(DATA_FOLDER, 'train')
TRAIN_DIR = os.path.join(DATA_FOLDER, 'training')
TEST_DIR = os.path.join(DATA_FOLDER, 'testing')
ALL_DATA = os.path.join(DATA_FOLDER, 'all_data')

TRAIN_CROPPED = os.path.join(DATA_FOLDER, 'train_cropped')
TEST_CROPPED = os.path.join(DATA_FOLDER, 'test_cropped')

if not os.path.isdir(TRAIN_CROPPED):
    os.makedirs(TRAIN_CROPPED)

if not os.path.isdir(TEST_CROPPED):
    os.makedirs(TEST_CROPPED)


# remove any non directories in RAW_DATA folder
# rename
# os.rename(os.path.join(RAW_DATA, 'ClientRaw'), os.path.join(RAW_DATA, 'real'))
# os.rename(os.path.join(RAW_DATA, 'ImposterRaw'), os.path.join(RAW_DATA, 'fake'))

# initial_raw_data = os.listdir(RAW_DATA)
# for f in initial_raw_data:
#     pf = os.path.join(RAW_DATA, f)
#     if not os.path.isdir(pf):            
#         os.remove(pf)

# raw1 = os.path.join(DATA_FOLDER, 'raw', 'real')
# raw0 = os.path.join(DATA_FOLDER, 'raw', 'fake')


# initial_dirs1 = os.listdir(raw1)
# initial_dirs0 = os.listdir(raw0)

# for raw1_dir in initial_dirs1:
#     path_dir = os.path.join(raw1, raw1_dir)
#     if os.path.isdir(path_dir) and not str(raw1_dir).endswith('.db'):
#         dirf.copy_directories(path_dir, raw1, copy=False, filter_directories=lambda x: not x.endswith('.db'))   
#     else:
#         os.remove(path_dir)
#     # remove the directory at the end
#     shutil.rmtree(path_dir)
        
# for raw0_dir in initial_dirs0:
#     path_dir = os.path.join(raw0, raw0_dir)

#     if os.path.isdir(path_dir) and not str(raw0_dir).endswith('.db'):
#         dirf.copy_directories(path_dir, raw0,copy=False, filter_directories=lambda x: not x.endswith('.db'))
#     else:
#         os.remove(path_dir)
#     shutil.rmtree(path_dir)


# dirf.dataset_portion(RAW_DATA, TRAIN_DIR, portion=0.9, copy=False)
# dirf.copy_directories(RAW_DATA, TEST_DIR, copy=False)



def set_up():
    # if os.path.isdir(RAW_DATA):
    #     shutil.rmtree(RAW_DATA)
    
    # dirf.unzip_data_file(os.path.join(DATA_FOLDER, 'data.zip'))
    # dirf.squeeze_directory(RAW_DATA)

    if os.path.isdir(TRAIN_DIR): 
        shutil.rmtree(TRAIN_DIR)

    if os.path.isdir(TEST_DIR):
        shutil.rmtree(TEST_DIR)

    if os.path.isdir(ALL_DATA):
        shutil.rmtree(ALL_DATA)
    
    os.makedirs(TRAIN_DIR)
    os.makedirs(TEST_DIR)
    
    os.makedirs(os.path.join(ALL_DATA, 'live'), exist_ok=True)
    os.makedirs(os.path.join(ALL_DATA, 'spoof'), exist_ok=True)

    for client_dir in os.listdir(RAW_DATA):    
        client_dir = os.path.join(RAW_DATA, client_dir)
        # copy the 'live' part to the all data 'live' part
        try:
            dirf.copy_directories(os.path.join(client_dir, 'live'), os.path.join(ALL_DATA, 'live'), copy=True)
            dirf.copy_directories(os.path.join(client_dir, 'spoof'), os.path.join(ALL_DATA, 'spoof'), copy=True)
        except Exception:
            # this probably means one folder has only 'live' or 'spoof'
            pass
        
# make a sanity check portion of the training dataset
# dirf.dataset_portion(TRAIN_DIR, 
#                     os.path.join(DATA_FOLDER, 'debug_portion'),
#                     portion=0.2, 
#                     copy=True)

from src.pytorch_modular.data_loaders import create_dataloaders
from FaceSpoofing.PixBiSupervision.training import preprocessing as pr, losses as ls
from  FaceSpoofing.PixBiSupervision.model import PiBiNet as binet,  DenseNetFE as dense
import src.pytorch_modular.image_classification.engine_classification as cls

from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import torch
from torch import nn

def binary_output(y_pred: Tuple[torch.Tensor, torch.Tensor]):
    _, logits = y_pred
    sigma = nn.Sigmoid()
    output = sigma(logits).to(torch.int32).squeeze()
    
    if torch.all(output == 1):
        print(f"all {1}")
 
    if torch.all(output == 0):
        print(f"all {0}")
 
    return output

def compute_loss(y_pred: torch.Tensor, 
                 y: torch.Tensor,
                 alpha: float=0.25) -> torch.Tensor:
    bfm, logits = y_pred
    feature_map_loss = ls.pixelBinaryLoss()(bfm, y.float())
    binary_loss = nn.BCEWithLogitsLoss()(logits.squeeze(), y.float())
    # return binary_loss
    return alpha * feature_map_loss + binary_loss * (1 - alpha)


def main(set_data: bool = False,
        train_dir=TRAIN_CROPPED, 
        test_dir=TEST_CROPPED):
    if set_data:
        set_up()

    preprocess = transforms.Compose([
        transforms.Resize((160, 160)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # prepare the data loaders
    train_dl, test_dl, _ = create_dataloaders(train_dir=train_dir, 
                                            test_dir=test_dir,
                                            train_transform=preprocess,
                                            batch_size=32)
    
    model = binet.PiBiNet(num_blocks=3)
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
                            'loss_function': nn.BCEWithLogitsLoss(),
                            'no_improve_stop': 10,
                            'output_layer': binary_output, 
                            'debug': False,
                            }

    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    results = cls.train_model(model, 
                            train_dataloader=train_dl, 
                            test_dataloader=test_dl, 
                            train_configuration=train_configuration,          
                            log_dir=os.path.join(HOME, 'runs'),         
                            save_path=os.path.join(HOME, 'saved_models'))   

# set_up()
# dirf.dataset_portion(directory_with_classes=ALL_DATA, 
#                     destination_directory=TRAIN_DIR,
#                     portion=0.9, 
#                     copy=False)

# dirf.copy_directories(src_dir=ALL_DATA, 
#                       des_dir=TEST_DIR, copy=False)

# pr.prepare_data(TRAIN_DIR, des_data_dir=TRAIN_CROPPED, limit=10 ** 4)
# pr.prepare_data(TEST_DIR, des_data_dir=TEST_CROPPED, limit = 4 * 10 ** 3)

# pr.prepare_data(TRAIN_DIR)
# pr.prepare_data(TEST_DIR)


# for clsdir in os.listdir(TRAIN_DIR):
#     counter = 0
#     for file_name in os.listdir(os.path.join(TRAIN_DIR, clsdir)):
#         if not file_name.startswith('cropped'):
#             os.remove(os.path.join(TRAIN_DIR, clsdir, file_name))

# min_count = min([len(os.listdir(os.path.join(TRAIN_DIR, d))) for d in os.listdir(TRAIN_DIR)])
# print(min_count)

# for clsdir in os.listdir(TRAIN_DIR):
#     for index, file_name in enumerate(os.listdir(os.path.join(TRAIN_DIR, clsdir))):
#         if index > min_count:
#             os.remove(os.path.join(TRAIN_DIR, clsdir, file_name))



if __name__ == '__main__':
    main()
    # remove any non_cropped image    
    # for clsdir in os.listdir(TEST_DIR):
    #     counter = 0
    #     for file_name in os.listdir(os.path.join(TEST_DIR, clsdir)):
    #         counter += int(file_name.startswith('cropped'))
    #     print(counter)
    