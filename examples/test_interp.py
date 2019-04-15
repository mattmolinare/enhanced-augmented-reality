#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

# local imports
import context
import ear

if __name__ == '__main__':

    # inputs
    input_image = r'../images/opencv.png'
    input_video = r'../videos/ps3-4-a/tmp.mp4'
    output_video = r'../videos/ps3-4-a/frame_step.mp4'
    num_features = 5000
    num_matches = 2000
    ransac_thresh = 10.0
    frame_step = 20
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
#    bbox = ear.get_corners(img.shape[0], img.shape[1])

    # allocate frames
    frames = np.empty((frame_step, rows, cols, 3), dtype=np.uint8)

    with ear.VideoWriter(output_video, fps, (cols, rows)) as writer:

        # write initial frame
        frame = ear.project_image(img, a, Ha)
#        frame = a.copy()
#        ear.draw_bbox(frame, ear.apply_homography(bbox, Ha))
        writer.write(frame)

        for frame_count, b in enumerate(gen, 1):

            print(frame_count)

            i = (frame_count - 1) % frame_step

            # pocket frame
            frames[i] = b

            if i != frame_step - 1:
                continue

            # update cumulative homography
            Hab = ear.get_homography(a, b, num_features, num_matches,
                                     ransac_thresh)
            Hb = ear.update_homography(Ha, Hab)

            # compute partial homographies
            homographies = ear.interpolate_homographies(Ha, Hb, frame_step,
                                                        method='direct')

            for b, Hb in zip(frames, homographies):
                # write frames
                frame = ear.project_image(img, b, Hb)
#                frame = b.copy()
#                ear.draw_bbox(frame, ear.apply_homography(bbox, Hb))
                writer.write(frame)

            # update a
            a[:] = b
            Ha[:] = Hb

            if frame_count == 1140:
                break
