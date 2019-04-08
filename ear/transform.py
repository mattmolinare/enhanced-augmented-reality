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

    x = np.ones(pts.shape[:-1] + (3,))
    x[..., :2] = pts

    x_tr = np.einsum('...ij,...j', H, x)
    pts_tr = x_tr[..., :2] / x_tr[..., 2, np.newaxis]

    return pts_tr
