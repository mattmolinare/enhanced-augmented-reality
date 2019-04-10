#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import context
import ear

if __name__ == '__main__':

    in_fname = r'..\videos\ps3-4-a\ps3-4-a.mp4'
    out_fname = r'..\videos\ps3-4-a\test.mp4'
    scale = 1.0 / 1

    bbox = np.array([
        [134.0, 108.0],
        [447.0, 109.0],
        [133.0, 250.0],
        [446.0, 256.0]
    ]) * scale

    num_features = 10000
    num_matches = 3000
    ransac_thresh = 5.0
    d_iter = 20
    max_iter = 10000

    props = ear.get_video_properties(in_fname)
    fps = props['fps'] / float(d_iter)
    gen = ear.generate_frames(in_fname)
    bgr1 = next(gen)
    bgr1 = ear.rescale_image(bgr1, scale)
    gray1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2GRAY)
    rows, cols = gray1.shape

    num_iter = 0
    H_list = []
    bbox_list = []
    n_list = []

    with ear.VideoWriter(out_fname, fps, (cols, rows)) as writer:

        out = bgr1.copy()
        ear.draw_bbox(out, bbox, thickness=2)
        writer.write(out)

        for bgr2 in gen:

            bgr2 = ear.rescale_image(bgr2, scale)
            gray2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY)

            num_iter += 1

            if num_iter % d_iter != 0:
                continue

            print(num_iter)

            # detect features
            kp1, kp2, desc1, desc2 = ear.feature_detectors \
                .orb_detector(gray1, gray2, num_features=num_features)

            # match features
            matches = ear.feature_matchers.bf_matcher(desc1, desc2)
            top_matches = matches[:num_matches]
            pts1, pts2 = ear.feature_matchers \
                .get_matched_points(kp1, kp2, top_matches)

            # compute homography
            H, mask = ear.transform \
                .compute_homography(pts1, pts2, ransac_thresh)
            H_list.append(H)
            n_list.append(mask.sum())

            # apply homography
            bbox = ear.transform.apply_homography(bbox, H)
            bbox_list.append(bbox)

            # write frame
            out = bgr2.copy()
            ear.draw_bbox(out, bbox, thickness=2)
            writer.write(out)

            # update frame1
            bgr1 = bgr2
            gray1 = gray2

            if num_iter > max_iter:
                break
