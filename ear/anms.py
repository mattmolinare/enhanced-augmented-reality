# -*- coding: utf-8 -*-

import numpy as np

__all__ = ['anms']


def anms(kpts):
    """Adaptive Non-Maximal Suppression
    """
    n = len(kpts)
    pts = np.empty((n, 2))
    responses = np.empty(n)
    for i, kpt in enumerate(kpts):
        pts[i] = kpt.pt
        responses[i] = kpt.response

    # compute pairwise distances
    dist = np.linalg.norm(pts - pts[:, np.newaxis], ord=2, axis=2) \
        .view(np.ma.MaskedArray)
    dist.mask = responses <= responses[:, np.newaxis]

    # sort by distance to nearest neighbor of greater Harris response
    min_dist = dist.min(axis=1)
    indices = min_dist.argsort()[::-1]

    return indices
