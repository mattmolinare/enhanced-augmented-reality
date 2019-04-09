# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot

__all__ = [
    'draw_rectangle',
    'easy_imshow',
    'rescale_image'
]


def easy_imshow(img, num=None, **imshow_kwargs):
    """Helper for ``matplotlib.pyplot.imshow`` for grayscale and BGR images
    """
    defaults = {
        'cmap': 'gray',
        'interpolation': 'none'
    }

    for key, val in defaults.items():
        imshow_kwargs[key] = imshow_kwargs.get(key, val)

    img = np.asarray(img)
    if img.ndim == 3:
        img = img[:, :, ::-1]

    fig = pyplot.figure(num)
    fig.clf()
    ax = fig.gca()
    ax.imshow(img, **imshow_kwargs)

    return fig, ax


def draw_rectangle(img, bbox, color=(0, 0, 255), thickness=1):
    """Bounding box is defineid as [top-left, top-right, bottom-left,
    bottom-right].
    """
    bbox = [tuple(pt) for pt in bbox.round().astype(np.int32)]

    def draw_line(i1, i2):
        cv2.line(img, bbox[i1], bbox[i2], color, thickness)

    draw_line(0, 1)
    draw_line(1, 3)
    draw_line(3, 2)
    draw_line(2, 0)


def rescale_image(image, scale):
    """Rescale image, preserving aspect ratio
    """
    return cv2.resize(image, None, fx=scale, fy=scale)
