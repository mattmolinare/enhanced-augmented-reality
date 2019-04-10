# -*- coding: utf-8 -*-

import cv2
import numpy as np

# local imports
from . import feature_matchers
from . import feature_detectors
from . import utils

__all__ = [
    'apply_homography',
    'compute_homography',
    'get_homography',
    'get_initial_homography',
    'project_image',
    'update_homography'
]


def compute_homography(pts1, pts2, ransac_thresh=None):
    """`pts1` and `pts2` have shape (n, 2)
    """
    if ransac_thresh is None:
        kwargs = dict(method=0)
    else:
        kwargs = dict(method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)

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


def get_homography(img1, img2, num_features, num_matches, ransac_thresh):

    # detect features
    kp1, kp2, desc1, desc2 = feature_detectors.orb_detector(
        img1, img2, num_features=num_features)

    # match features
    matches = feature_matchers.bf_matcher(desc1, desc2)[:num_matches]
    pts1, pts2 = feature_matchers.get_matched_points(kp1, kp2, matches)

    # estimate homography
    H, _ = compute_homography(pts1, pts2, ransac_thresh=ransac_thresh)

    return H


def update_homography(H1, H2):

    H = H2.dot(H1)
    H *= 1.0 / H[2, 2]

    return H
