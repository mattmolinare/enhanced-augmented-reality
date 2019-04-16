#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

# local imports
import context
import ear

if __name__ == '__main__':

    fname1 = r'..\videos\kitchen\frames\frame0101.png'
    fname2 = r'..\videos\kitchen\frames\frame0202.png'

    num_features = 2000
    num_matches = 200
    num_anms = 100
    ransac_thresh = 5.0

    img1 = cv2.imread(fname1, 0)
    img2 = cv2.imread(fname2, 0)

    kpts1, kpts2, desc1, desc2 = ear.orb_detector(img1, img2,
                                                  num_features=num_features)
    matches = ear.bf_matcher(desc1, desc2)[:num_matches]

    matched_kpts1 = [kpts1[match.queryIdx] for match in matches]
    anms_indices = ear.anms(matched_kpts1)[:num_anms]
    anms_matches = np.take(matches, anms_indices)

    pts1, pts2 = ear.get_matched_points(kpts1, kpts2, anms_matches)
    H, ransac_mask = ear.compute_homography(pts1, pts2, ransac_thresh)
    ransac_matches = np.compress(ransac_mask[:, 0], anms_matches)

    print('# bf matches: %i' % len(matches))
    print('# ransac matches: %i' % len(ransac_matches))

    res = cv2.drawMatches(img1, kpts1, img2, kpts2, ransac_matches, None,
                          flags=2)
    ear.easy_imshow(res, num=1)
