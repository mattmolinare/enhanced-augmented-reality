# -*- coding: utf-8 -*-

import cv2
import numpy as np

__all__ = [
    'apply_homography',
    'compute_homography'
]


def compute_homography(pts1, pts2, ransac_thresh):
    return cv2.findHomography(pts1, pts2, method=cv2.RANSAC,
                              ransacReprojThreshold=ransac_thresh)


def apply_homography(pts, H):
    return cv2.perspectiveTransform(pts[np.newaxis], H)[0]
