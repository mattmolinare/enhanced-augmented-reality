# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot

__all__ = [
    'compute_edge_gradient',
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


def get_corners(rows, cols):
    """Get coordinates of image corners
    """
    corners = np.array([[0, 0], [cols, 0], [0, rows], [cols, rows]],
                       dtype=np.float64)

    return corners


def compute_edge_gradient(img, ksize=5):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)

    grad = np.abs(grad_x) + np.abs(grad_y)

    return grad


def get_initial_bbox(img, frame):
    pass

    # return get_corners(frame)
    # bbox = np.array([
    #     [134.0, 108.0],
    #     [447.0, 109.0],
    #     [133.0, 250.0],
    #     [446.0, 256.0]
    # ])

    # return bbox
