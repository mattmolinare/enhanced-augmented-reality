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
    num_features = 10000
    num_matches = 3000
    ransac_thresh = 3.0
    frame_step = 1

    # file io
    img = cv2.imread(input_image)
    fps = ear.get_fps(input_video)
    gen = ear.generate_frames(input_video)

    # initialize
    a = next(gen)
    rows, cols, _ = a.shape

    A = np.empty((frame_step, rows, cols, 3), dtype=np.uint8)
    H = np.empty((frame_step, 3, 3))

    A[0] = a
    H[0] = ear.get_initial_homography(img, a)

    for i in range(1, frame_step):

        b = next(gen)
        H_ab = ear.get_homography(a, b, num_features, num_matches,
                                  ransac_thresh)

        A[i] = b
        H[i] = ear.update_homography(H[0], H_ab)

    with ear.VideoWriter(output_video, fps, (cols, rows)) as writer:

        for b, H_i in zip(A, H):
            # write initial frames
            frame = ear.project_image(img, b, H_i)
            writer.write(frame)

        for frame_count, b in enumerate(gen, frame_step):

            print(frame_count)
            i = frame_count % frame_step

            a = A[i]
            H_i = H[i]

            # update cumulative homography
            H_ab = ear.get_homography(a, b, num_features, num_matches,
                                      ransac_thresh)
            H_i[:] = ear.update_homography(H_i, H_ab)

            # write frame
            frame = ear.project_image(img, b, H_i)
            writer.write(frame)

            # update A
            a[:] = b
