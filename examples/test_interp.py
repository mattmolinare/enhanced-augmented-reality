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
    num_frames = 300
    ransac_thresh = 10.0
    frame_step = 5

    # file io
    img = cv2.imread(input_image)
    fps = ear.get_fps(input_video)
    gen = ear.generate_frames(input_video)

    # initialize
    a = next(gen)
    rows, cols, _ = a.shape
    Ha = ear.get_initial_homography(img, a)

    # allocate frames
    frames = np.empty((frame_step, rows, cols, 3), dtype=np.uint8)

    with ear.VideoWriter(output_video, fps, (cols, rows)) as writer:

        # write initial frame
        frame = ear.project_image(img, a, Ha)
        writer.write(frame)

        for frame_count, b in enumerate(gen, 1):

            print(frame_count)

            i = (frame_count - 1) % frame_step

            # pocket frame
            frames[i] = b

            if i != frame_step - 1:
                continue

            # estimate homography
            Hab = ear.get_homography(a, b, num_features, num_matches,
                                     ransac_thresh)
            Hb = ear.update_homography(Ha, Hab)

            # estimate partial homographies
            homographies = ear.interpolate_homographies(Ha, Hb, frame_step,
                                                        method='direct')

            for b, Hb in zip(frames, homographies):
                # write frames
                frame = ear.project_image(img, b, Hb)
                writer.write(frame)

            # update a
            a[:] = b
            Ha[:] = Hb

            if frame_count > num_frames:
                break
