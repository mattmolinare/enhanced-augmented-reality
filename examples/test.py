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
    output_video = r'..\videos\ps3-4-a\test.mp4'
    num_features = 5000
    num_matches = 2000
    ransac_thresh = 10.0
#    median_size = 11
#    sobel_size = 3
#    edge_thresh = 40

    # file io
    img = cv2.imread(input_image)
    fps = ear.get_fps(input_video)
    gen = ear.generate_frames(input_video)

    # initialize
    a = next(gen)
    rows, cols = a.shape[:2]
#    bbox = ear.get_initial_bbox(img, a, median_size, sobel_size, edge_thresh)
    bbox = np.array([
        [134.0, 108.0],
        [447.0, 109.0],
        [133.0, 250.0],
        [446.0, 256.0]
    ])
    Ha = ear.get_initial_homography(img, bbox)

    with ear.VideoWriter(output_video, fps, (cols, rows)) as writer:

        # write initial frame
        frame = ear.project_image(img, a, Ha)
        writer.write(frame)

        for frame_count, b in enumerate(gen, 1):

            print(frame_count)

            # update cumulative homography
            Hab = ear.get_homography(a, b, num_features, num_matches,
                                     ransac_thresh)
            Hb = ear.update_homography(Ha, Hab)

            # write frame
            frame = ear.project_image(img, b, Hb)
            writer.write(frame)

            # update a
            a[:] = b
            Ha[:] = Hb
