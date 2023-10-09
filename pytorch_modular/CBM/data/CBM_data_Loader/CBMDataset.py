"""
This script contains functionalities designed to load data efficiently
"""
import os
import clip
import torch

from torchvision.datasets import ImageFolder
from typing import Union, List
from pathlib import Path

from pytorch_modular import directories_and_files as dirf
from pytorch_modular.CBM.data.labels_generation.label_creator import CBMLabelGenerator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ConceptDataset(ImageFolder):

    def _prepare_labels(self):
        # the idea here is to iterate through the self.root directory
        initial_walk_through = os.walk(self.root, topdown=True)

        for root, dirs, files in initial_walk_through:
            if len(files) > 0:
                # first create a new folder to save the label of the images in the corresponding class
                os.makedirs(os.path.join(self.root, f'{root}_labels'), exist_ok=True)
                # the next step is to iterate through every image and create its soft labels
                # we can create the label for each

    def __init__(self,
                 root: Union[str, Path],
                 concepts: List[str],
                 label_generator: CBMLabelGenerator = None):

        root = dirf.process_save_path(root, file_ok=False, dir_ok=True,
                                      # make sure that all the sub files are indeed directories
                                      condition=lambda x: all([os.path.isdir(os.path.join(x, p))
                                                               for p in os.listdir(x)]))
        # convert to root for the super class
        root = str(root)
        super().__init__(root)

        # initialize a label generator
        self.label_generator = CBMLabelGenerator()
        # save the features of the concepts as they will be used for the label generation of every image
        self.concepts_features = self.label_generator.encode_concepts(concepts)
