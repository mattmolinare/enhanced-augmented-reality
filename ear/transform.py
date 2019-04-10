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
    'interpolate_homographies',
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


def interpolate_homographies(H1, H2, num_steps, method='direct'):
    """Result has shape (`num_steps`, 3, 3)
    """
    if num_steps == 1:
        return H2

    homographies = np.empty((num_steps, 3, 3))
    homographies[-1] = H2

    if method == 'direct':
        delta = (H2 - H1) / num_steps
        homographies[:-1] = H1 + \
            delta * np.arange(1, num_steps)[:, np.newaxis, np.newaxis]

    elif method == 'polar_decomp':
        raise NotImplementedError

    else:
        raise ValueError('no such method: %s' % method)

    return homographies


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


def update_homography(H, H12):
    """`H` is cumulative homography at image 1 and `H12` is homography between
    image 1 and image 2
    """
    H = H12.dot(H)
    H *= 1.0 / H[2, 2]

    return H
