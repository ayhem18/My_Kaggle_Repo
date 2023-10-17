"""
This script contains functionalities designed to efficiently load data for Concept Bottleneck Models
"""

import itertools
import math
import os
import torch
import shutil

import torchvision.transforms as tr

from torch.utils.data import Dataset
from typing import Union, List, Dict, Tuple
from pathlib import Path
from PIL import Image

from pytorch_modular import directories_and_files as dirf
from pytorch_modular.CBM.data.labels_generation.Clip_label_generation import ClipLabelGenerator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ConceptDataset(Dataset):
    concept_label_ending = 'concept_label'

    @classmethod
    def load_sample(cls, sample_path: Union[str, Path]):
        # make sure the path is absolute
        if not os.path.isabs(sample_path):
            raise ValueError(f"The loader is expecting the sample path to be absolute.\nFound:{sample_path}")

        # this code is copied from the DatasetFolder python file
        with open(sample_path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def find_classes(self) -> Tuple[List[str], Dict[str, int]]:
        """
        This function iterates through the 'self.root' directory returning a list of classes
        and mapping each class to a number

        Returns: the list of classes and a string-to-index map
        """

        index_counter = 0
        cls_index_map = {}
        classes = []

        folder_names = sorted(os.listdir(self.root))
        for cls in folder_names:
            # make sure to ignore folder that end with the concept label ending
            if not cls.endswith(self.concept_label_ending):
                cls_index_map[cls] = index_counter
                index_counter += 1
                classes.append(cls)
        return classes, cls_index_map

    def _sample_to_concept_label(self, sample_path: str, cls_label: str = None) -> str:
        """
        Each sample is associated with an embedding representing the distance between the picture
        and the given concept. Such embedding is saved in the disk to avoid going through the inference process
        each time during training.

        Args:
            sample_path: the path to the sample
            cls_label: the label of the given sample. passed for debugging purposes

        Returns: a path where the corresponding label tensor should be saved

        """

        # make sure the path to the sample is absolute
        if not os.path.isabs(sample_path):
            raise ValueError(f"The path is expected to be absolute. Found: {sample_path}")

        extracted_label = os.path.basename(Path(sample_path).parent)

        # this is used mainly for debugging purposes
        if cls_label is not None and extracted_label != cls_label:
            raise ValueError(f"expected the class labels to match. Given: {cls_label}, Retrieved: {extracted_label}")

        # extract the name of the file
        sample_name, _ = os.path.splitext(os.path.basename(sample_path))
        # build the path to the tensor label
        tensor_path = os.path.join(self.root, f'{extracted_label}_{self.concept_label_ending}', f'{sample_name}.pt')
        return tensor_path

    def _prepare_labels(self, batch_size: int = 64, debug: bool = False) -> None:
        """
        This function iterates through the 'self.root' directory creating the concepts labels for each sample
        and saving them in separate directories within 'self.root'. Such process is conducted to avoid
        repeated inference during training.

        Returns:
        """
        initial_walk_through = os.walk(self.root, topdown=True)

        for root, _, files in initial_walk_through:
            # non-empty directories are assumed to be data directories

            if not root.endswith(self.concept_label_ending) and len(files) > 0:

                self.data_count += len(files)
                # convert the file name to is absolute form
                files = [os.path.join(root, f) for f in files]

                # at this point of the code, 'root' should represent an absolute path to a class folder
                cls_name = os.path.basename(root)

                # first create a new folder to save the label of the images in the corresponding class
                os.makedirs(os.path.join(self.root, f'{cls_name}_{self.concept_label_ending}'), exist_ok=True)

                # calculate the number of batches
                num_batches = math.ceil(len(files) / batch_size)

                for i in range(num_batches):
                    # first generate the labels
                    batch_labels = (self.label_generator.generate_image_label(
                        files[i * batch_size: (i + 1) * batch_size], self.concepts_features))

                    # the 'batch_labels' object represent a 2d tensor where each row represents the sample's label.
                    for label_index, label_tensor in enumerate(batch_labels):
                        file_path = files[i * batch_size + label_index]
                        torch.save(label_tensor,
                                   self._sample_to_concept_label(sample_path=file_path, cls_label=cls_name))

    def _cls_to_range(self) -> Dict[str, Tuple[int, int]]:
        """
        This method builds the tools needed to efficiently map a numerical index to a unique sample absolute path
        """
        # build a mapping from classes to range of indices
        range_min = 0
        cls_range_map = {}
        for cls in self.classes:
            folder_files = len(os.listdir(os.path.join(self.root, cls)))
            cls_range_map[cls] = (range_min, range_min + folder_files - 1)
            range_min += folder_files

        return cls_range_map

    def __init__(self,
                 root: Union[str, Path],
                 concepts: Union[Dict[str, List[str]], List[str]],
                 image_transform: tr = None,
                 label_generator=None,
                 remove_existing: bool = True, 
                 debug: bool = False):
        """
        Args:
            root: the root directory
            concepts: a list / dictionary of concepts used for all classes
            image_transform: transformation applied on a given image
            remove_existing: whether to remove already-existing concept directories
        """
        # the default transformation is converting to torch.Tensors
        if image_transform is None:
            # the main idea here is to resize the images to a small size to be able to stack them into a single tensor
            image_transform = tr.Compose([tr.Resize(size=(224, 224)), tr.ToTensor()])

        self.root = dirf.process_save_path(root,
                                           file_ok=False,
                                           dir_ok=True,
                                           # make sure that all the sub files are indeed directories
                                           condition=lambda x: all([os.path.isdir(os.path.join(x, p))
                                                                    for p in os.listdir(x)]),
                                           error_message=f'The root directory is expected to have only directories as'
                                                         f' inner files.')
        # process the concepts
        concepts = concepts if isinstance(concepts, list) else list(itertools.chain(*(concepts.values())))
        # filter duplicate concepts
        self.concepts = list(set(concepts))

        self.image_transform = image_transform
        # a variable to save the number of samples in the dataset
        self.data_count = 0

        # remove existing concept labels if needed
        if remove_existing:
            for inner_dir in os.listdir(root):
                if inner_dir.endswith(self.concept_label_ending):
                    shutil.rmtree(os.path.join(root, inner_dir))

        # save the number of initial folders in 'root'
        initial_num_folders = len(os.listdir(self.root))

        # build the class to index map following Pytorch API
        self.classes, self.class_to_idx = self.find_classes()

        # build a mapping between classes and the associated range of indices
        self.index_to_class = self._cls_to_range()

        # create the label generator
        self.label_generator = label_generator if label_generator is not None else ClipLabelGenerator()
        # save the features of the concepts as they will be used for the label generation of every image
        self.concepts_features = self.label_generator.encode_concepts(concepts)
        self._prepare_labels(debug=debug)

        # at this point the number of directories should have doubled
        assert not remove_existing or len(os.listdir(root)) == 2 * initial_num_folders, \
            "The number of directories is not doubled !!!"

    def __getitem__(self, index: int):
        # first step is to locate the path to the corresponding sample
        sample_path, sample_cls = None, None
        for cls, indices_range in self.index_to_class.items():
            if indices_range[0] <= index <= indices_range[1]:
                sample_cls = cls
                cls_dir = os.path.join(self.root, cls)
                sample_path = os.path.join(cls_dir, os.listdir(cls_dir)[index - indices_range[0]])
                break

        sample_image = self.load_sample(sample_path)
        # apply the transformation
        sample_image = self.image_transform(sample_image) if self.image_transform is not None else sample_image
        # convert the class as 'str' to an index
        cls_label = self.class_to_idx[sample_cls]

        # retrieve the concept label
        concept_label = torch.load(self._sample_to_concept_label(sample_path))

        return sample_image, concept_label, cls_label

    def __len__(self) -> int:
        return self.data_count
