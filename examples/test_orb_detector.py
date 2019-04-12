#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

# local imports
import context
import ear

if __name__ == '__main__':

    fname1 = r'..\videos\ps3-4-a\frames\frame0101.png'
    fname2 = r'..\videos\ps3-4-a\frames\frame1002.png'

    num_features = 5000
    num_matches = 2000
    ransac_thresh = 10.0

    img1 = cv2.imread(fname1, 0)
    img2 = cv2.imread(fname2, 0)

    kp1, kp2, desc1, desc2 = ear.orb_detector(img1, img2,
                                              num_features=num_features)
    matches = ear.bf_matcher(desc1, desc2)[:num_matches]
    pts1, pts2 = ear.get_matched_points(kp1, kp2, matches)
    H, mask = ear.compute_homography(pts1, pts2, ransac_thresh)
    ransac_matches = np.compress(mask[:, 0], matches)

    print('# bf matches: %i' % len(matches))
    print('# ransac matches: %i' % len(ransac_matches))

    res = cv2.drawMatches(img1, kp1, img2, kp2, ransac_matches, None, flags=2)
    ear.easy_imshow(res, num=1)
