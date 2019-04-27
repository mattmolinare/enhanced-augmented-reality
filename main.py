#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import sys

# local import
import ear

default_params = []

default_params.append({
    'input_image': r'images/opencv.png',
    'input_video': r'videos/kitchen/kitchen_720p.mp4',
    'output_prefix': r'output/default/default_kitchen_720p',
    'num_features': 2000,
    'num_matches': 2000,
    'num_anms': None,
    'ransac_thresh': 20.0,
    'frame_step': 8,
    'median_size': 11,
    'sobel_size': 3,
    'edge_thresh': 40,
    'verbose': False
})

default_params.append({
    'input_image': r'images/opencv.png',
    'input_video': r'videos/living_room/living_room_720p.mp4',
    'output_prefix': r'output/default/default_living_room_720p',
    'num_features': 2000,
    'num_matches': 2000,
    'num_anms': None,
    'ransac_thresh': 20.0,
    'frame_step': 8,
    'median_size': 11,
    'sobel_size': 3,
    'edge_thresh': 40,
    'verbose': False
})

default_params.append({
    'input_image': r'images/opencv.png',
    'input_video': r'videos/office/office.mp4',
    'output_prefix': r'output/default/default_office',
    'num_features': 5000,
    'num_matches': 3000,
    'num_anms': None,
    'ransac_thresh': 20.0,
    'frame_step': 8,
    'median_size': 11,
    'sobel_size': 3,
    'edge_thresh': 40,
    'verbose': False
})


def main(params):

    # file io
    if not os.path.isfile(params['input_video']):
        raise FileNotFoundError('no such input video: %s'
                                % params['input_video'])

    img = cv2.imread(params['input_image'])
    fps = ear.get_fps(params['input_video'])
    gen = ear.generate_frames(params['input_video'])

    output_dir = os.path.dirname(params['output_prefix'])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    _, ext = os.path.splitext(params['input_video'])
    output_video = params['output_prefix'] + ext

    # initialize
    a = next(gen)
    rows, cols = a.shape[:2]
    bbox = ear.get_initial_bbox(img, a, params['median_size'],
                                params['sobel_size'], params['edge_thresh'])
    Ha = ear.get_initial_homography(img, bbox)

    # allocate frame cache
    frames = np.empty((params['frame_step'], rows, cols, 3), dtype=np.uint8)

    with ear.VideoWriter(output_video, fps, (cols, rows)) as writer:

        msg = 'writing to %s...' % output_video
        print(msg)

        # write initial frame
        frame = ear.project_image(img, a, Ha)
        writer.write(frame)

        for frame_count, b in enumerate(gen, 1):

            if params['verbose']:
                print('on frame %i' % frame_count)

            i = (frame_count - 1) % params['frame_step']

            # pocket frame
            frames[i] = b

            if i != params['frame_step'] - 1:
                continue

            # update cumulative homography
            Hab = ear.get_homography(a, b, params['num_features'],
                                     params['num_matches'],
                                     params['ransac_thresh'],
                                     num_anms=params.get('num_anms'))
            Hb = ear.update_homography(Ha, Hab)

            # compute partial homographies
            homographies = ear.interpolate_homographies(Ha, Hb,
                                                        params['frame_step'],
                                                        method='direct')

            for b, Hb in zip(frames, homographies):
                # write frames
                frame = ear.project_image(img, b, Hb, copy=True)
                writer.write(frame)

            # update a
            a[:] = b
            Ha[:] = Hb

            if 'num_frames' in params and frame_count > params['num_frames']:
                break

        print(msg + 'done!')


if __name__ == '__main__':

    try:
        import yaml
        has_yaml = True
    except ImportError:
        has_yaml = False

    if len(sys.argv) > 1:
        fname = sys.argv[1]
        if has_yaml:
            with open(fname, 'r') as fp:
                params = yaml.load(fp, Loader=yaml.FullLoader)
        else:
            print('yaml library is unavailable; using default parameters')
            params = default_params
    else:
        print('no yaml file provided; using default parameters')
        params = default_params

    if not isinstance(params, list):
        params = [params]

    for p in params:
        with ear.Profiler(['time'], [10]) as pf:
            main(p)
        pf.dump_stats(p['output_prefix'] + '.stats')
        if has_yaml:
            yaml_fname = p['output_prefix'] + '.yaml'
            if not os.path.isfile(yaml_fname):
                with open(yaml_fname, 'w') as fp:
                    yaml.dump(p, fp)
