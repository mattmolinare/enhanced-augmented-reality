#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

# local imports
import context
import ear

if __name__ == '__main__':

    fname1 = r'..\videos\living_room\frames\frame0101.png'
    fname2 = r'..\videos\living_room\frames\frame0601.png'

    num_features = 5000
    num_matches = 2000
    num_anms = 30
    ransac_thresh = 20.0

    bgr1 = cv2.imread(fname1)
    bgr2 = cv2.imread(fname2)

    img1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY)

    img1 = cv2.resize(img1, None, fx=1/4, fy=1/4)
    img2 = cv2.resize(img2, None, fx=1/4, fy=1/4)

    kpts1, kpts2, desc1, desc2 = ear.orb_detector(img1, img2,
                                                  num_features=num_features)
    matches = ear.bf_matcher(desc1, desc2)[:num_matches]

    matched_kpts1 = [kpts1[match.queryIdx] for match in matches]
    anms_indices = ear.anms(matched_kpts1)[:num_anms]
    anms_matches = np.take(matches, anms_indices)

    pts1, pts2 = ear.get_matched_points(kpts1, kpts2, anms_matches)
    H, ransac_mask = ear.compute_homography(pts1, pts2, ransac_thresh)
    ransac_matches = anms_matches[ransac_mask]

    print('# bf matches: %i' % len(matches))
    print('# ransac matches: %i' % len(ransac_matches))

    res1 = cv2.drawMatches(img1, kpts1, img2, kpts2, ransac_matches, None,
                           flags=2)
    ear.easy_imshow(res1, num=1)

    res2 = np.dstack((img1,) * 3)
    for pt in pts1[ransac_mask]:
        cv2.circle(res2, tuple(pt), 10, (255, 0, 0), 3)
    ear.easy_imshow(res2, num=2)
