import os
import requests
import numpy as np
import torch
import cv2 as cv

from pathlib import Path
from typing import Union, List, Dict, Tuple
from _collections_abc import Sequence
from collections import defaultdict
from ultralytics import YOLO
from ultralytics.engine.results import Results

from face.custom_tracker import CustomByteTracker
from face.utilities import FR_SingletonInitializer


def images_to_numpy(images: Sequence[Union[str, Path]]) -> List[np.ndarray]:
    if isinstance(images, (str, Path)):
        images = [images]
    return [np.asarray(cv.imread(img)) for img in images]


def xywh_converter(xywh: Sequence[int, int, int, int]) -> \
        Tuple[int, int, int, int]:
    # the idea is to accept a xywh bounding box and return 4 coordinates that can directly be
    # used for cropping
    x_center, y_center, w, h = xywh
    return y_center - h // 2, y_center + h // 2, x_center - w // 2, x_center + w // 2


def crop_image(image: np.ndarray,
               bb_coordinates: Tuple[int, int, int, int],
               debug: bool = False) -> np.ndarray:
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    y0, y1, x0, x1 = bb_coordinates
    cropped_image = image[y0: y1, x0: x1, :]
    if debug:
        cv.imshow('original', image)
        cv.imshow('cropped', cropped_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return cropped_image


class YoloAnalyzer(object):
    _tracker_URL = 'https://raw.githubusercontent.com/voyager-108/ml/main/apps/nn/yolov8/trackers/tracker.yaml'

    @classmethod
    def _file_dir(cls):
        try:
            return os.path.realpath(os.path.dirname(__file__))
        except NameError:
            return os.getcwd()

    @classmethod
    def _tracker_file(cls):
        # download the tracker file as needed
        req = requests.get(cls._tracker_URL, allow_redirects=True)
        with open(os.path.join(cls._file_dir(), 'tracker.yaml'), 'wb') as f:
            f.write(req.content)

    @classmethod
    def frames_signature(cls,
                         tracking_results: List[Results],
                         xywh: bool = False):
        """
        This function returns a unique signature for each frame: ([list of bounding boxes], [list of ids])
        Args:
            tracking_results: This a list of Results objects
            xywh: whether to return the bounding box in the xywh format or not

        Returns:
        """

        def signature(tres_obj: Results) -> Tuple[List, List]:
            if tres_obj.boxes is None:
                return [], []
            boxes = [b.xywh.squeeze().cpu().tolist() if xywh else xywh_converter(b.xywh.squeeze().cpu().tolist()) 
                    for b in tres_obj.boxes.cpu()]
            
            ids = [b.id.int().tolist() for b in tres_obj.boxes.cpu()]
            return boxes, ids

        return [signature(tres_obj) for tres_obj in tracking_results]

    def __init__(self,
                 top_persons_detected: int = 5,
                 top_faces_detected: int = 2,
                 yolo_path: Union[str, Path] = 'yolov8n.pt',
                 ) -> None:
        self.top_persons_detected = top_persons_detected
        # the top_faces_detected is preferably odd
        self.top_faces_detected = top_faces_detected + int(top_faces_detected % 2 == 0)

        # download the 'tracker.yaml' file if needed
        if not os.path.isfile(os.path.join(self._file_dir(), 'tracker.yaml')):
            # download the file in this case
            self._tracker_file()

        # the yolo components
        self.yolo = YOLO(yolo_path)
        self.tracker = CustomByteTracker(os.path.join(self._file_dir(), 'tracker.yaml'))

        singleton = FR_SingletonInitializer()
        self.face_detector = singleton.get_face_detector()
        self.face_encoder = singleton.get_encoder()
        self.device = singleton.get_device()

    def _track(self, frames: Sequence[Union[Path, str, np.ndarray, torch.Tensor]]) -> List[Results]:
        """This function tracks the different people detected across the given sequence of frames

        Args:
            frames (Sequence[Union[Path, str, np.ndarray, torch.Tensor]]): a sequence of frames. The assumption is
            that frames are consecutive in time.

        Returns:
            List[Results]: a list of Yolo Results objects
        """
        # this is the first step is the face detection pipeline: Detecting people in the image + tracking them
        tracking_results = self.yolo.track(source=frames,
                                           persist=True,
                                           classes=[0],  # only detect people in the image
                                           device=self.device,
                                           show=False)

        # remove the extra ids by calling the custom tracker's method
        self.tracker.track(tracking_results)
        return tracking_results

    def _identify(self, tracking_results: List[Results]):
        # create a dictionary to save the information about each id detected in the results
        # the dictionary will be of the form {id: [(frame_index, boxes, probs)]}
        ids_dict = defaultdict(lambda: [])

        # iterate through the results to extract the ids
        for frame_index, results in enumerate(tracking_results):

            boxes = results.boxes

            if boxes is None:
                continue
            
            ids = boxes.id.int().cpu().tolist()
            probs = boxes.conf


            assert len(ids) == len(boxes) == len(probs), "Check the lengths of ids, probabilities and boxes"

            for i, bb, p in zip(ids, boxes.xywh.cpu().tolist(), probs):
                # convert the xywh bounding box to coordinates that can be used directly for image cropping
                bounding_coordinates = xywh_converter(bb)
                ids_dict[i].append((frame_index, bounding_coordinates, p))

        # the final step is to filter the results
        # keep only the top self.person_detected boxes for each id
        for person_id, info in enumerate(ids_dict):
            ids_dict[person_id] = sorted(info, key=lambda x: x[-1], reverse=True)[:self.top_persons_detected]

        return ids_dict

    def _detect_encode(self, frames,
                       person_dict: Dict[int, List],
                       crop_person: bool = True,
                       debug: bool = False) -> Dict[int, List[np.ndarray]]:

        if crop_person:
            frames = images_to_numpy(frames)

        # detect only the most probable face, as the input will be cropped to contain only one image
        self.face_detector.keep_all = False
        # iterate through the person_dict
        id_cropped_images = dict([(person_id, [crop_image(frames[frame_index], bb, debug=debug)
                                               for frame_index, bb, _ in person_info])
                                  for person_id, person_info in person_dict.items()])
        # for each id get the faces in each of the cropped images
        id_faces = dict([(person_id, self.face_detector.forward(cropped_images, return_prob=True))
                         for person_id, cropped_images in id_cropped_images.items()])

        # for each id keep only the best self.person_detect images
        for person_id, face_prob in id_faces.items():
            # sort the faces according to their probabilities
            sorted_output = sorted(face_prob, key=lambda x: x[-1], reverse=True)[:self.top_faces_detected]
            faces = [f for f, p in sorted_output]
            # encode the output
            id_faces[person_id] = self.face_encoder(faces).cpu().numpy() if len(faces) != 0 else []

        return id_faces

    def analyze_frames(self,
                       frames: Sequence[Union[Path, str, np.ndarray, torch.Tensor]],
                       xywh: bool = False,
                       debug: bool = False):
        # the first step is to track the frames
        tracking_results = self._track(frames)
        # create a signature for each frame
        frame_signatures = self.frames_signature(tracking_results=tracking_results)
        # group each 'person' identified in the picture
        person_ids = self._identify(tracking_results=tracking_results)
        # time to encode each person ('id') detected by YOLO
        face_ids = self._detect_encode(frames, person_ids, xywh=xywh, debug=debug)

        return frame_signatures, face_ids
