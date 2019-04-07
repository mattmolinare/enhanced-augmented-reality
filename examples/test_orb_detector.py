#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import context
import ear

if __name__ == '__main__':

    fname1 = r'C:\Users\mattm\Documents\gatech\spring2019\cs6476\ear\videos\office\frames8\frame0001.png'
    fname2 = r'C:\Users\mattm\Documents\gatech\spring2019\cs6476\ear\videos\office\frames8\frame0011.png'

    num_matches = 20
    ransac_thresh = 5.0

    img1 = cv2.imread(fname1, 0)
    img2 = cv2.imread(fname2, 0)

    kp1, kp2, desc1, desc2 = ear.feature_detectors.orb_detector(img1, img2)
    matches = ear.feature_matchers.bf_matcher(desc1, desc2, num_matches)
    pts1, pts2 = ear.feature_matchers.get_matched_points(kp1, kp2, matches)
    H = ear.transform.compute_homography(pts1, pts2,
                                         ransac_thresh=ransac_thresh)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
    ear.easy_imshow(img3, num=1)
