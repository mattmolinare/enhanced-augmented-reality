#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

# local imports
import context
import ear

if __name__ == '__main__':

    # inputs
    input_image = r'..\images\opencv.png'
    input_video = r'..\videos\ps3-4-a\ps3-4-a.mp4'
    output_video = r'..\videos\ps3-4-a\test2.mp4'
    num_features = 10000
    num_matches = 3000
    ransac_thresh = 3.0
    frame_step = 5

    # file io
    img = cv2.imread(input_image)
    fps = ear.get_fps(input_video)
    gen = ear.generate_frames(input_video)

    # initialize
    frame_a = next(gen)
    rows, cols = frame_a.shape[:2]
    bbox_a = ear.get_initial_bbox(frame_a)
    H = ear.get_initial_homography(img, bbox_a)
    skipped_frames = np.empty((frame_step - 1,) + frame_a.shape,
                              dtype=np.uint8)
    frame_counter = 0

    with ear.VideoWriter(output_video, fps, (cols, rows)) as vw:

        frame_out = ear.project_image(img, frame_a, H)
        vw.write(frame_out)

        for frame_b in gen:

            print(frame_counter)

            frame_counter += 1
            mod = frame_counter % frame_step

            if mod != 0:
                skipped_frames[mod - 1] = frame_b
                continue

            # detect features
            kp_a, kp_b, desc_a, desc_b = ear.orb_detector(
                frame_a, frame_b, num_features=num_features)

            # match features
            matches = ear.bf_matcher(desc_a, desc_b)[:num_matches]
            pts_a, pts_b = ear.get_matched_points(kp_a, kp_b, matches)

            # compute homography
            H_ab, _ = ear.compute_homography(pts_a, pts_b,
                                             ransac_thresh=ransac_thresh)

            # update bounding box
            bbox_b = ear.apply_homography(bbox_a, H_ab)

            # get shift in location of bounding box corners
            dbbox = (bbox_b - bbox_a) / frame_step

            H_i = H
            for i in range(frame_step - 1):

                # linearly interpolate corners of bounding box
                frame_i = skipped_frames[i]
                bbox_i = bbox_a + dbbox * (i + 1)

                # compute "partial" homography
                H_ai, _ = ear.compute_homography(bbox_a, bbox_i)
                H_i = ear.update_homography(H, H_ai)

                # write frame
                frame_out = ear.project_image(img, frame_i, H_i)
                vw.write(frame_out)

            # update cumulative homography
            H = ear.update_homography(H, H_ab)

            # write frame
            frame_out = ear.project_image(img, frame_b, H)
            vw.write(frame_out)

            frame_a = frame_b
            bbox_a = bbox_b
