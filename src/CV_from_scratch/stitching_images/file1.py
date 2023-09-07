"""
This script is taken from the following tutorial:
https://pyimagesearch.com/2016/01/11/opencv-panorama-stitching/
"""
import os

import numpy as np
import cv2

from _collections_abc import Sequence


# noinspection PyMethodMayBeStatic,PyUnresolvedReferences
class Stitcher:
    def __init__(self, affine_transform: bool = True):
        self.affine = affine_transform

    def stitch(self,
               images: Sequence[np.array, np.array],
               ratio=0.75,
               reproj_thresh=4.0,
               show_matches=False, ):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detect_and_describe(imageA)
        (kpsB, featuresB) = self.detect_and_describe(imageB)

        # match the points between the 2 images
        matches = self.match_keypoints(kpsA,
                                       kpsB,
                                       featuresA,
                                       featuresB,
                                       ratio,
                                       reproj_thresh)
        if M is None:
            return None
        # calculate the transformation matrix
        result = cv2.warpPerspective(imageA, H,
                                     (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))

        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        # check to see if the keypoint matches should be visualized
        if show_matches:
            vis = self.draw_matches(imageA,
                                    imageB,
                                    kpsA,
                                    kpsB,
                                    matches,
                                    status)
            # return a tuple of the stitched image and the
            # visualization
            return result, vis
        # return the stitched image
        return result

    def detect_and_describe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect and extract features from the image
        descriptor = cv2.SIFT_create()
        kps, features = descriptor.detectAndCompute(image, None)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])
        # return a tuple of keypoints and features
        return kps, features

    def match_keypoints(self,
                        kpsA,
                        kpsB,
                        featuresA,
                        featuresB,
                        ratio,
                        reprojThresh) -> Optional[Tuple[List[Tuple[int, int]]]]:
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        min_matches = 3 if self.affine else 3

        if len(matches) >= 2:
            im1_points = np.array([kpsA[match_point_1] for match_point_1, match_point_2 in matches],
                                  dtype=np.float32)
            im2_points = np.array([kpsB[match_point_2] for match_point_1, match_point_2 in matches],
                                  dtype=np.float32)
            return im1_points, im2_points

        return None

    def draw_matches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB
        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        # return the visualization
        return vis


if __name__ == '__main__':
    p1, p2 = os.path.join('data', 'original', '__1.jpg'), os.path.join('data', 'original', '__2.jpg')

    im1, im2 = cv2.imread(p1), cv2.imread(p2)
    # stitch the images together to create a panorama
    stitcher = Stitcher()
    (result, vis) = stitcher.stitch([im1, im2], show_matches=True)
    # show the images
    # cv2.imshow("Image A", im1)
    # cv2.imshow("Image B", im2)
    # cv2.imshow("Keypoint Matches", vis)
    # cv2.imshow("Result", result)
    cv2.waitKey(0)
