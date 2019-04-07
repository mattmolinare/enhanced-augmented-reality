# -*- coding: utf-8 -*-

import cv2

__all__ = [
    'compute_homography'
]


def compute_homography(pts1, pts2, ransac_thresh=5.0):
    H, _ = cv2.findHomography(pts1, pts2, method=cv2.RANSAC,
                              ransacReprojThreshold=ransac_thresh)
    return H
