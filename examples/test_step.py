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
    frame_step = 20

    # file io
    img = cv2.imread(input_image)
    fps = ear.get_fps(input_video)
    gen = ear.generate_frames(input_video)

    # initialize
    a = next(gen)
    rows, cols, _ = a.shape
    Ha = ear.get_initial_homograpy(img, a)

    # populate arrays
    frames = np.empty((frame_step, rows, cols, 3), dtype=np.uint8)
    homographies = np.empty((frame_step, 3, 3))

    frames[0] = a
    homographies[0] = Ha

    for i in range(1, frame_step):

        b = next(gen)
        Hab = ear.get_homography(a, b, num_features, num_matches,
                                 ransac_thresh)
        Hb = ear.update_homography(Ha, Hab)

        frames[i] = b
        homographies[i] = Hb

    with ear.VideoWriter(output_video, fps, (cols, rows)) as writer:

        for b, Hb in zip(frames, homographies):
            # write initial frames
            frame = ear.project_image(img, b, Hb)
            writer.write(frame)

        for frame_count, b in enumerate(gen, frame_step):

            print(frame_count)

            i = frame_count % frame_step

            a = frames[i]
            Ha = homographies[i]

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
