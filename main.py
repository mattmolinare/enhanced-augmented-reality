#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import glob
import numpy as np
import sys

# local import
import ear


def main(fname):

    params = ear.read_yaml(fname)

    img = cv2.imread(params['image_in'])
    props = ear.get_video_properties(params['video_in'])
    gen = ear.generate_frames(params['video_in'])
    video_writer = ear.VideoWriter(params['video_out'], props['fps'],
                                   (props['cols'], props['rows']))

    frame_in1 = next(gen)
    bbox = ear.get_initial_bbox(frame_in1)
    H = ear.get_initial_homography(img, frame_in1)
    i = 1

    with video_writer as writer:

        # project image into first frame
        frame_out = ear.project_image(img, frame_in1, H)
        if params['draw_bbox']:
            ear.draw_bbox(frame_out, bbox, thickness=2)

        # write first frame
        if params['verbose']:
            print('writing frame %i' % i)
        writer.write(frame_out)

        for frame_in2 in gen:

            i += 1

            # detect features
            kp1, kp2, desc1, desc2 = ear.orb_detector(
                frame_in1, frame_in2, num_features=params['num_features'])

            # match features
            matches = ear.bf_matcher(desc1, desc2)[:params['num_matches']]
            pts1, pts2 = ear.get_matched_points(kp1, kp2, matches)

            # compute homography
            H_12, mask = ear.compute_homography(
                pts1, pts2, ransac_thresh=params['ransac_thresh'])

            if params['verbose']:
                print('\t# matches: %i' % len(matches))
                print('\t# inliers: %i' % np.count_nonzero(mask))

            # update cumulative homography
            H = H_12.dot(H)
            H /= H[2, 2]

            # project image into frame
            frame_out = ear.project_image(img, frame_in2, H)
            if params['draw_bbox']:
                bbox = ear.apply_homography(bbox, H_12)
                ear.draw_bbox(frame_out, bbox, thickness=2)

            # write frame
            if params['verbose']:
                print('writing frame %i' % i)
            writer.write(frame_out)

            # update frame
            frame_in1 = frame_in2


if __name__ == '__main__':

    if len(sys.argv) > 1:
        fname = sys.argv[1]
    else:
        for fname in glob.iglob('*.yaml'):
            break
        else:
            raise IOError('no .yaml file found')

    main(fname)
