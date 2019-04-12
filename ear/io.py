# -*- coding: utf-8 -*-

import cv2

__all__ = [
    'VideoReader',
    'VideoWriter',
    'generate_frames',
    'get_fps',
]


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

    codecs = {
     'mp4': 'MP4V'
    }

    def __init__(self, fname, fps, size):

        ext = fname.rsplit('.', 1)[1].lower()
        if ext not in VideoWriter.codecs:
            raise ValueError('invalid file name extension: %s' % ext)

        self.fname = fname
        self.fourcc = cv2.VideoWriter_fourcc(*VideoWriter.codecs[ext])
        self.fps = fps
        self.size = tuple(size)

    def __enter__(self):
        self._writer = cv2.VideoWriter(self.fname, self.fourcc, self.fps,
                                       self.size)
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
