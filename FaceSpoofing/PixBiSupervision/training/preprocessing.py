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
                 des_data_dir: Union[str, Path],
                 batch_size: int = 256,
                 limit = 10 ** 5,
                 device: str = None):
    
    os.makedirs(des_data_dir, exist_ok=True)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    # initialize the face detection model
    face_detector = MTCNN(image_size=160,  # needed for th model later
                          keep_all=False,
                          post_process=True,
                          # we want all the returned images to be the same size
                          select_largest=True,
                          device=device).eval()

    assert len(os.listdir(data_dir)) == 2
    for cls_dir in os.listdir(data_dir):
        
        des_data_dir = os.path.join(des_data_dir, cls_dir)
        os.makedirs(des_data_dir, exist_ok=True)

        cls_dir = os.path.join(data_dir, cls_dir)
        # extract the file names first
        images = [os.path.join(cls_dir, im) for im in os.listdir(cls_dir)]
        counter = 0
        while counter < min(len(images), limit):
            batch = [Image.open(str(img)) if isinstance(img, (str, Path)) else img
                     for img in images[counter: counter + batch_size]]
            # remove the original images
            # for im in images[counter: counter + batch_size]:
            #     os.remove(im)
            try:
                boxes = face_detector.detect(batch) 
            except Exception:
                boxes = [face_detector.detect(im) for im in batch]
                # if for any reason the batch preprocessing did not work.
                # encoded_ims = [face_detector.forward(im) for im in batch]

            # save the data 
            for index, (img, box) in enumerate(zip(batch, boxes),start=counter):
                img = np.asarray(img)
                # print(box)
                if box is None: 
                    print("no faces detected")
                    continue

                box, _ = box
                try:
                    box = np.squeeze(box)
                    if len(box.shape) > 1:
                        box = box[0]
                    box = [int(b) for b in box]      
                except:
                    print("more than 1 face detected for some reason")
                    continue

                cropped_img = img[max(0, box[1]):box[3], max(0, box[0]):box[2], :]
                face_save_path = os.path.join(des_data_dir, f'cropped_image_{index}.jpg')
                cv.imwrite(filename = face_save_path, img=cropped_img)

                # if img is not None:
                #     save_image(img, face_save_path)
            counter += batch_size            
            print(f'batch {counter // batch_size + 1} processed !!')
        