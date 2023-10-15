"""This script contains functionalities to train the Optimal Sequential Grouping algorithm.
"""
import torch
import OSG
import numpy as np
import osg_vsd_dataset
import OptimalSequentialGrouping
import os

from torch.utils.data import DataLoader
from pytorch_modular import directories_and_files as dirf, pytorch_utilities as ut

np.set_printoptions(linewidth=300)


def train_model(dir_path, num_iterations: int = 101, stop_parameter: float = 0.75):
    # let's make sure we have the data we need
    dir_path = dirf.process_save_path(save_path=dir_path, 
                                      dir_ok=True, 
                                      file_ok=False, 
                                      condition=(lambda path: 
                                      all([f.endswith('.hdf5') for f in os.listdir(path)]) and os.path.basename(path).endswith('visual')))
    
    # let's define the parameters: 
    d, K_max = 2048, 5
    BN = True # batch normalization
    DO = 0.0 # the dropout rate
    dist_metric = 'cosine'
    dist_type = 'EMBEDDING'
    feature_sizes = [d, 3000, 3000, 1000, 100] # the number of hidden units in the linear network
    learning_rate = 0.005
    weight_decay = 1e-2

    # after making sure the data is as expected, let's proceed
    device = ut.get_default_device()
    
    # define the dataset and the dataloader
    dataset = osg_vsd_dataset.VSD_DATASET(data_dir_path=dir_path)

    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    # define the model
    osg_model = OSG.OSG_C(feature_sizes, K_max, BN, DO, dist_type, dist_metric, device)

    