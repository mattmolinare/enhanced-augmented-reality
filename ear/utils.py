# -*- coding: utf-8 -*-

import cProfile
import cv2
import numpy as np
import pstats


__all__ = [
    'compute_edge_gradient',
    'draw_bbox',
    'easy_imshow',
    'get_corners',
    'get_initial_bbox',
    'rescale_image',
    'Profiler'
]


class Profiler:

    def __init__(self, keys=[], restrictions=[]):
        self.keys = keys
        self.restrictions = restrictions

    def __enter__(self):
        self._pf = cProfile.Profile()
        self._pf.enable()
        return self._pf

    def __exit__(self, type, value, traceback):
        self._pf.disable()
        p = pstats.Stats(self._pf)
        p.sort_stats(*self.keys)
        p.print_stats(*self.restrictions)


def easy_imshow(img, num=None, **imshow_kwargs):
    """Helper for ``matplotlib.pyplot.imshow`` for grayscale and BGR images
    """
    from matplotlib import pyplot

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


def draw_bbox(img, bbox, color=(255, 255, 0), thickness=3):
    """Bounding box is defined as [top-left, top-right, bottom-left,
    bottom-right].
    """
    bbox = [tuple(pt) for pt in bbox.round().astype(np.int32)]

    def draw_line(i, j):
        cv2.line(img, bbox[i], bbox[j], color, thickness)

    draw_line(0, 1)
    draw_line(1, 3)
    draw_line(3, 2)
    draw_line(2, 0)


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


def compute_edge_gradient(img, sobel_size=3):

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    Gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_size)
    Gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_size)

    G = np.abs(Gx) + np.abs(Gy)

    return G


def get_initial_bbox(img, frame, median_size, sobel_size, edge_thresh):
    """Get initial bounding box using the maximum distance to an edges in
    `frame`. Bounding box is defined as [top-left, top-right, bottom-left,
    bottom-right].
    """
    # apply median filter
    frame_blur = cv2.medianBlur(frame, median_size)

    # compute gradient
    grad = compute_edge_gradient(frame_blur, sobel_size=sobel_size)

    # apply threshold on gradient to get non-edges
    non_edges = np.where(grad > edge_thresh, 0, 255).astype(np.uint8)

    # compute l2-distance to any edge
    non_edges_pad = np.pad(non_edges, (1, 1), mode='constant',
                           constant_values=0)
    dist = cv2.distanceTransform(non_edges_pad, cv2.DIST_L2, 5,
                                 dstType=cv2.CV_32F)[1:-1, 1:-1]

    # get largest circle in non-edge region
    y, x = divmod(dist.argmax(), dist.shape[1])
    r = dist[y, x]

    # compute largest bounding box that can fit inside the circle
    tan = img.shape[0] / img.shape[1]
    dx = r / (1 + tan ** 2) ** 0.5
    dy = dx * tan

    bbox = np.array([
        [x - dx, y - dy],
        [x + dx, y - dy],
        [x - dx, y + dy],
        [x + dx, y + dy]
    ])

    return bbox
