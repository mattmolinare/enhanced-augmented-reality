# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot

__all__ = [
    'easy_imshow'
]


def easy_imshow(img, num=None, **imshow_kwargs):

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
