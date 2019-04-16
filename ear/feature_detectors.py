# -*- coding: utf-8 -*-

import cv2

__all__ = [
    'akaze_detector',
    'orb_detector',
]


def orb_detector(img1, img2, num_features=500):

    orb = cv2.ORB_create(nfeatures=num_features)

    kpts1, desc1 = orb.detectAndCompute(img1, None)
    kpts2, desc2 = orb.detectAndCompute(img2, None)

    return kpts1, kpts2, desc1, desc2


def akaze_detector(img1, img2):

    akaze = cv2.AKAZE_create()

    kpts1, desc1 = akaze.detectAndCompute(img1, None)
    kpts2, desc2 = akaze.detectAndCompute(img2, None)

    return kpts1, kpts2, desc1, desc2
