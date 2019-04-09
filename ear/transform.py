# -*- coding: utf-8 -*-

import cv2
import numpy as np

# local imports
from . import utils

__all__ = [
    'apply_homography',
    'compute_homography',
    'get_initial_homography',
    'project_image'
]


def compute_homography(pts1, pts2, ransac_thresh=None):
    """`pts1` and `pts2` have shape (n, 2)
    """
    if ransac_thresh is None:
        # use least squares
        kwargs = {'method': 0}
    else:
        # use RANSAC
        kwargs = {
            'method': cv2.RANSAC,
            'ransacReprojThreshold': ransac_thresh
        }

    return cv2.findHomography(pts1, pts2, **kwargs)


def apply_homography(pts, H):
    """`pts` has shape (n, 2)
    """
    return cv2.perspectiveTransform(pts[np.newaxis], H)[0]


def project_image(img, frame, H):
    """Project image into frame using homography
    """
    frame = frame.copy()
    rows, cols = frame.shape[:2]
    cv2.warpPerspective(img, H, (cols, rows), dst=frame,
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_TRANSPARENT)

    return frame


def get_initial_homography(img, frame):

    pts1 = utils.get_corners(img)
    pts2 = utils.get_initial_bbox(frame)

    H, _ = compute_homography(pts1, pts2)

    return H
