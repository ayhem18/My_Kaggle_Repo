"""
This script contains functionalities designed to efficiently load data for Concept Bottleneck Models
"""
import math
import os
import torch
import itertools
import shutil

from torchvision.datasets import ImageFolder
from typing import Union, List, Dict
from pathlib import Path

from pytorch_modular import directories_and_files as dirf
from pytorch_modular.CBM.data.labels_generation.Clip_label_generation import ClipLabelGenerator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ConceptDataset(ImageFolder):
    def _label_tensor_path(self, cls_label: str, sample_path: str):
        if not os.path.isabs(sample_path):
            raise ValueError(f"The path is expected to be absolute. Found: {sample_path}")

        extracted_label = os.path.basename(Path(sample_path).parent)

        if extracted_label != cls_label:
            raise ValueError(f"expected the class labels to match. Given: {cls_label}, Retrieved: {extracted_label}")

        # extract the name of the file
        sample_name = os.path.basename(sample_path)
        # split the name and the extension
        sample_name, _ = os.path.splitext(sample_path)
        # build the path to the tensor label
        tensor_path = os.path.join(self.root, f'{cls_label}_labels', f'{sample_name}.pt')
        return tensor_path

    def _prepare_labels(self, batch_size: int = 64) -> None:
        # the idea here is to iterate through the self.root directory
        initial_walk_through = os.walk(self.root, topdown=True)

        for root, _, files in initial_walk_through:
            if len(files) > 0:
                # convert the file name to absolute paths
                files = [os.path.join(root, f) for f in files]

                # first create a new folder to save the label of the images in the corresponding class
                os.makedirs(os.path.join(self.root, f'{root}_labels'), exist_ok=True)

                num_batches = math.ceil(len(files) / batch_size)

                for i in range(num_batches):
                    # first generate the labels
                    batch_labels = (self.label_generator.generate_image_label(
                        files[i * batch_size: (i + 1) * batch_size], self.concepts_features))
                    # at this point the labels for each image in the batch were generated: time to save them
                    for label_index, label_tensor in enumerate(batch_labels):
                        file_path = files[i * batch_size + label_index]
                        torch.save(label_tensor, self._label_tensor_path(cls_label=os.path.basename(root), sample_path=file_path))

    def custom_loader(self, sample_path: Union[str, Path]):
        sample_path = str(sample_path) if isinstance(sample_path, Path) else sample_path
        # first use the parent loader to load the image and the class
        loader_output = self.loader(sample_path)
        image, class_label = loader_output
        # convert the image to the absolute path if needed
        sample_path = sample_path if os.path.isabs(sample_path) else os.path.join(self.root, class_label, sample_path)
        # retrieve the concept label
        concept_label = torch.load(self._label_tensor_path(sample_path))
        return image, concept_label, class_label

    def __init__(self,
                 root: Union[str, Path],
                 concepts: Union[Dict[str, List[str]], List[str]]):

        root = dirf.process_save_path(root, file_ok=False, dir_ok=True,
                                      # make sure that all the sub files are indeed directories
                                      condition=lambda x: all([os.path.isdir(os.path.join(x, p))
                                                               for p in os.listdir(x)]))
        
        for dir in os.listdir(root):
            if dir.endswith('labels'):
                shutil.rmtree(os.path.join(root, dir))
        
        initial_num_folders = len(os.listdir(root))

        root = str(root)
        super().__init__(root)

        # extract the concepts
        concepts = list(set(itertools.chain(*(concepts.values()))))

        # initialize a label generator
        self.label_generator = ClipLabelGenerator()
        # save the features of the concepts as they will be used for the label generation of every image
        self.concepts_features = self.label_generator.encode_concepts(concepts)
        self._prepare_labels()
        # at this point the number of directories should have doubled
        assert len(os.listdir(root)) == 2 * initial_num_folders, "The number of directories is not doubled !!!"

        # the next step is to override the parent loader
        self.loader = lambda sample_path: self.custom_loader(sample_path=sample_path)

    