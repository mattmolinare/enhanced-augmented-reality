#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

# local imports
import context
import ear

if __name__ == '__main__':

    fname = r'../images/test.png'
    k_median = 3
    k_sobel = 5
    q = 95.0
    rows = 100
    cols = 200

    img = cv2.imread(fname)
    img_filt = cv2.medianBlur(img, k_median)
    grad = ear.compute_edge_gradient(img_filt, ksize=k_sobel)
    thresh = np.percentile(grad, q)
    edges = np.where(grad > thresh, 0, 255).astype(np.uint8)
    edges_pad = np.pad(edges, (1, 1), mode='constant', constant_values=0)
    dist = cv2.distanceTransform(edges_pad, cv2.DIST_L2, 5, dstType=cv2.CV_32F)[1:-1, 1:-1]
    y, x = idx = divmod(dist.argmax(), img.shape[1])
    max_dist = dist[idx]

    aspect_ratio = rows / cols
    width = max_dist / np.sqrt(1 + aspect_ratio ** 2.0)
    height = width * aspect_ratio

    bbox = np.array([
        [x - width, y - height],
        [x + width, y - height],
        [x - width, y + height],
        [x + width, y + height]
    ])

    ear.draw_bbox(img, bbox, (0, 255, 0), 2)
    cv2.circle(img, (x, y), max_dist, (0, 0, 255), 2)

    fig = plt.figure(1)
    fig.clf()
    axes = fig.subplots(2, 2)
    axes[0, 0].imshow(img[:, :, ::-1])
    axes[0, 1].imshow(grad, cmap='gray')
    axes[1, 0].imshow(edges, cmap='gray_r')
    axes[1, 1].imshow(dist, cmap='jet')
