"""
"""
import os

import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as tr

from torchvision import datasets
from torch.utils.data import DataLoader

from pathlib import Path
from typing import Union

from facenet_pytorch import MTCNN
from FaceSpoofing.PixBiSupervision.model.PiBiNet import PiBiNet

from PIL import Image
from torchvision.utils import save_image

script_dir = os.path.dirname(os.path.realpath(__file__))

train_dir = os.path.join(script_dir, 'data', 'train')
test_dir = os.path.join(script_dir, 'data', 'test')


def prepare_data(data_dir: Union[str, Path],
                 batch_size: int = 256,
                 device: str = None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    # initialize the face detection model
    face_detector = MTCNN(image_size=160,  # needed for th model later
                          keep_all=False,
                          post_process=True,
                          # we want all the returned images to be the same size
                          select_largest=True,
                          # return the face with the highest probability, not the largest
                          device=device).eval()

    assert len(os.listdir(data_dir)) == 2
    for cls_dir in os.listdir(data_dir):
        cls_dir = os.path.join(data_dir, cls_dir)
        # extract the file names first
        images = [os.path.join(cls_dir, im) for im in os.listdir(cls_dir)]
        counter = 0
        while counter < len(images):
            batch = [Image.open(str(img)) if isinstance(img, (str, Path)) else img
                     for img in images[counter: counter + batch_size]]
            # remove the original images
            for im in images[counter: counter + batch_size]:
                os.remove(im)

            encoded_ims = face_detector.forward(batch,
                                                return_prob=False)
            
            # save the data 
            for index, img in enumerate(encoded_ims,start=counter):
                face_save_path = os.path.join(cls_dir, f'cropped_image_{index}.jpg')
                save_image(img, face_save_path)

            counter += batch_size            
            print(f'batch {counter // batch_size + 1} processed !!')

# def collate_preprocessing(batch) -> torch.Tensor:
#     images_as_tensors = tr.ToTensor()(batch)
#     faces_cropped = MTCNN.forward(images_as_tensors, return_prob=False)
#     # apply the PiBiNet default transformation
#     final_faces = PiBiNet.default_transformation(faces_cropped)
#     return final_faces


# def prepare_dataloaders(train_dir: Union[str, Path],
#                         test_dir: Union[str, Path],
#                         batch_size: int = 32):
#     # create the dataset first
#     train_data = datasets.ImageFolder(train_dir)
#     test_data = datasets.ImageFolder(test_dir)

#     train_dl = DataLoader(
#         train_data,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=os.cpu_count() // 2,
#         pin_memory=True,
#         drop_last=True,
#         collate_fn=collate_preprocessing)

#     test_dl = DataLoader(
#         test_data,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=os.cpu_count() // 2,
#         pin_memory=True,
#         drop_last=False,
#         collate_fn=collate_preprocessing)

#     return train_dl, test_dl
