# -*- coding: utf-8 -*-

import cv2
import numpy as np

__all__ = [
    'bf_matcher',
    'get_matched_points'
]


def get_matched_points(kp1, kp2, matches):

    pts1 = np.empty((len(matches), 2), dtype=np.float32)
    pts2 = np.empty_like(pts1)

    for i, match in enumerate(matches):
        pts1[i] = kp1[match.queryIdx].pt
        pts2[i] = kp2[match.trainIdx].pt

    return pts1, pts2


def bf_matcher(desc1, desc2):

    bf = cv2.BFMatcher_create(normType=cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(desc1, desc2)
    matches.sort(key=lambda x: x.distance)

    return matches
