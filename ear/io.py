# -*- coding: utf-8 -*-

import cv2
import yaml

__all__ = [
    'generate_frames',
    'read_yaml'
]


def read_yaml(fname):
    with open(fname, 'r') as fp:
        data = yaml.load(fp)
    return data


def generate_frames(fname):
    cap = cv2.VideoCapture(fname)
    while cap.isOpened():
        retval, frame = cap.read()
        if retval:
            yield frame
        else:
            break
    cap.release()
