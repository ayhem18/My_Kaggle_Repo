import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
torch.manual_seed(69)

from typing import Union, Any, Tuple
from torch.utils.data import Dataset
from pathlib import Path
import os
import sys


home = os.path.dirname(os.path.realpath(__file__))
current = home


TRAIN_DATA = os.path.join(home, 'data', 'train.csv')
TEST_DATA = os.path.join(home, 'data', 'test.csv')


while 'pytorch_modular' not in os.listdir(current):
    current = Path(current).parent

sys.path.append(str(current))
sys.path.append(os.path.join(str(current), 'pytorch_modular'))



import cv2 as cv

class TrainDs(Dataset):
    def __init__(self, train_data: Union[Path, str]) -> None:
        super().__init__()
        self.data = pd.read_csv(train_data, index_col=0)

    def __getitem__(self, index) -> Any:
        # get the image
        image_data, label = self.data.iloc[index, :-1].values, self.data.iloc[index, -1].item()
        assert isinstance(label, (int, float))
        # reshape the data and normalize
        image_data = image_data.reshape(28, 28, 1).astype(np.uint8)
        # convert to rgb
        image_data = cv.cvtColor(image_data, cv.COLOR_GRAY2BGR)
        # normalize
        image_data = image_data / 255.0

        return torch.from_numpy(image_data).permute(2, 0, 1).float(), label
    
    def __len__(self):
        return len(self.data) 


class TestDs(Dataset):
    def __init__(self, test_data: Union[Path, str]) -> None:
        super().__init__()
        self.data = pd.read_csv(test_data, index_col=0)

    def __getitem__(self, index) -> Any:
        # get the image
        image_data = self.data.iloc[index].values
    
        image_data = image_data.reshape(28, 28, 1).astype(np.uint8)
        # convert to rgb
        image_data = cv.cvtColor(image_data, cv.COLOR_GRAY2BGR)
        # normalize
        image_data = image_data / 255.0

        # make sure to permute the image dimensions to be compatible with pytorch
        return torch.from_numpy(image_data).permute(2, 0, 1).float()
           
    def __len__(self):
        return len(self.data) 
    


from torch.utils.data import DataLoader


from pytorch_modular.visual import display_image


from pytorch_modular.transfer_learning import resnetFeatureExtractor as rfe
from pytorch_modular.image_classification import classification_head as ch
from pytorch_modular.dimensions_analysis import dimension_analyser as da

from pytorch_modular.image_classification import engine_classification
from pytorch_modular.image_classification import epoch_engine
import pytorch_modular.exp_tracking as exp

from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch import nn


class ClassificationModel(nn.Module):
    def __init__(self, 
                 feature_extractor, 
                 classification_head: ch.GenericClassifier,
                 input_shape: Tuple[int, int, int], 
                 num_classes: int,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.fe = feature_extractor
        self.head = classification_head
        analyser = da.DimensionsAnalyser()   

        # extract the number of channels
        c_index = np.argmin(input_shape)
        batched_input = tuple([dim for i, dim in enumerate(input_shape) if i != c_index])
        batched_input = (1, ) + batched_input + (input_shape[c_index], )

        # find the number of input features
        in_features = analyser.analyse_dimensions_static(feature_extractor, input_shape=batched_input)
        
        in_features = np.prod(in_features)

        self.head.in_features = in_features
        self.head.num_classes = num_classes
        
        self.model = nn.Sequential(self.fe, nn.Flatten(), self.head)

    def forward(self, x: torch.Tensor):
        return self.model.forward(x)

from sklearn.model_selection import train_test_split
def solution():

    train_ds = TrainDs(TRAIN_DATA)
    train_ds, val_ds = train_test_split(train_ds, random_state=69, test_size=0.1)
    test_ds = TestDs(test_data=TEST_DATA)


    train_dataloader = DataLoader(
        train_ds,
        batch_size=512,
        shuffle=True,
        num_workers=os.cpu_count() // 2,
        pin_memory=True,
        drop_last=True,
    )

    test_dataloader = DataLoader(
        val_ds,
        batch_size=512,
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        pin_memory=True,
        drop_last=False,
    )


    feature_extractor = rfe.ResNetFeatureExtractor(num_blocks=1, freeze=False)
    head = ch.ExponentialClassifier(num_classes=10, in_features=1024, num_layers=4)
    model = ClassificationModel(feature_extractor=feature_extractor, classification_head=head, input_shape=(1, 28, 28), num_classes=10)

    optimizer = Adam(params=model.parameters(), lr=0.05)
    scheduler = LinearLR(optimizer=optimizer, start_factor=1, end_factor=0.1, total_iters=100) 

    train_configuration = {'optimizer': optimizer,
                            'scheduler': scheduler,
                            'min_val_loss': 10 ** -4,
                            'max_epochs': 10,
                            'report_epoch': 1,
                            }

    engine_classification.train_model(model=model, 
                                    train_dataloader=train_dataloader, 
                                    test_dataloader=test_dataloader, 
                                    train_configuration=train_configuration, 
                                    log_dir=os.path.join(home,'runs'),
                                    save_path=os.path.join(home, 'modelds'))

if __name__ == '__main__':
    solution()