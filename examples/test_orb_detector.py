#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

# local imports
import context
import ear

if __name__ == '__main__':

    fname1 = r'..\videos\office\frames\frame0101.png'
    fname2 = r'..\videos\office\frames\frame0801.png'

    num_features = 5000
    num_matches = 5000
    num_anms = 100
    ransac_thresh = 20.0

    img1 = ear.rescale_image(cv2.imread(fname1, 0), 0.5)
    img2 = ear.rescale_image(cv2.imread(fname2, 0), 0.5)

    kpts1, kpts2, desc1, desc2 = ear.orb_detector(img1, img2,
                                                  num_features=num_features)
    matches = ear.bf_matcher(desc1, desc2)[:num_matches]

    matched_kpts1 = [kpts1[match.queryIdx] for match in matches]
    anms_indices = ear.anms(matched_kpts1)[:num_anms]
    anms_matches = np.take(matches, anms_indices)

    pts1, pts2 = ear.get_matched_points(kpts1, kpts2, anms_matches)
    H, ransac_mask = ear.compute_homography(pts1, pts2, ransac_thresh)
    ransac_matches = anms_matches[ransac_mask]

    res1 = cv2.drawMatches(img1, kpts1, img2, kpts2, ransac_matches, None,
                           flags=2)
    ear.easy_imshow(res1, num=1)

    res2 = np.dstack((img1,) * 3)
    for pt in pts1[ransac_mask]:
        cv2.circle(res2, tuple(pt), 6, (255, 0, 0), 2)
    ear.easy_imshow(res2, num=2)

    print('# bf matches: %i' % len(matches))
    print('# ransac matches: %i' % len(ransac_matches))
