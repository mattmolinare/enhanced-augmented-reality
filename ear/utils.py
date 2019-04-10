# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot

__all__ = [
    'draw_bbox',
    'easy_imshow',
    'get_corners',
    'get_initial_bbox',
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


def draw_bbox(img, bbox, color=(0, 0, 255), thickness=1):
    """Bounding box is defined as [top-left, top-right, bottom-left,
    bottom-right].
    """
    bbox = [tuple(pt) for pt in bbox.round().astype(np.int32)]

    cv2.line(img, bbox[0], bbox[1], color, thickness)
    cv2.line(img, bbox[1], bbox[3], color, thickness)
    cv2.line(img, bbox[3], bbox[2], color, thickness)
    cv2.line(img, bbox[2], bbox[0], color, thickness)


def rescale_image(img, scale):
    """Rescale image, preserving aspect ratio
    """
    return cv2.resize(img, None, fx=scale, fy=scale)


def get_corners(img):
    """Get coordinates of image corners
    """
    rows, cols = img.shape[:2]
    corners = np.array([[0, 0], [cols, 0], [0, rows], [cols, rows]],
                       dtype=np.float64)

    return corners


def get_initial_bbox(frame):
    # TODO: MBS algorithm
#    return get_corners(frame)

    bbox = np.array([
        [134.0, 108.0],
        [447.0, 109.0],
        [133.0, 250.0],
        [446.0, 256.0]
    ])

    return bbox
