# -*- coding: utf-8 -*-

import cv2
import yaml

__all__ = [
    'VideoReader',
    'VideoWriter',
    'generate_frames',
    'get_fps',
    'read_yaml'
]


def read_yaml(fname):
    with open(fname, 'r') as fp:
        data = yaml.load(fp)
    return data


class VideoReader:

    def __init__(self, fname):
        self.fname = fname

    def __enter__(self):
        self._reader = cv2.VideoCapture(self.fname)
        return self._reader

    def __exit__(self, type, value, traceback):
        self._reader.release()
        del self._reader

    def __bool__(self):
        return self._reader.isOpened()


class VideoWriter:

    def __init__(self, fname, fps, size):
        self.fname = fname
        self.fps = fps
        self.size = tuple(size)

    def __enter__(self):
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self._writer = cv2.VideoWriter(self.fname, fourcc, self.fps, self.size)
        return self._writer

    def __exit__(self, type, value, traceback):
        self._writer.release()


def get_fps(fname):
    with VideoReader(fname) as reader:
        fps = reader.get(cv2.CAP_PROP_FPS)
    return fps


def generate_frames(fname):
    with VideoReader(fname) as reader:
        while reader:
            retval, frame = reader.read()
            if retval:
                yield frame
            else:
                break
