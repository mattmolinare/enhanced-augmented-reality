#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

# local imports
import context
import ear

if __name__ == '__main__':

    fname = r'../videos/office/frames/frame0001.png'
    median_size = 7
    sobel_size = 3
    edge_thresh = 40
    img_rows = 100
    img_cols = 100

    frame = cv2.imread(fname)
    frame = ear.rescale_image(frame, 1 / 2)
    frame_blur = cv2.medianBlur(frame, median_size)
    grad = ear.compute_edge_gradient(frame_blur, sobel_size=sobel_size)
    non_edges = np.where(grad > edge_thresh, 0, 255).astype(np.uint8)
    non_edges_pad = np.pad(non_edges, (1, 1), mode='constant',
                           constant_values=0)
    dist = cv2.distanceTransform(non_edges_pad, cv2.DIST_L2, 5,
                                 dstType=cv2.CV_32F)[1:-1, 1:-1]

    y, x = divmod(dist.argmax(), dist.shape[1])
    r = dist[y, x]

    tan = img_rows / img_cols
    dx = r / (1 + tan ** 2) ** 0.5
    dy = dx * tan

    bbox = np.array([
        [x - dx, y - dy],
        [x + dx, y - dy],
        [x - dx, y + dy],
        [x + dx, y + dy]
    ])

    ear.draw_bbox(frame, bbox, (0, 255, 0), 2)
    cv2.circle(frame, (x, y), r, (255, 0, 255), 2)

    fig = plt.figure(1)
    fig.clf()
    axes = fig.subplots(2, 2)
    axes[0, 0].imshow(frame[:, :, ::-1])
    axes[0, 1].imshow(grad, cmap='jet')
    axes[1, 0].imshow(non_edges, cmap='gray_r')
    axes[1, 1].imshow(dist, cmap='jet')
