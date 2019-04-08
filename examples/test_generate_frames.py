#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import context
import ear

if __name__ == '__main__':

    fname = r'..\videos\ps3-4-a\ps3-4-a.mp4'
    gen = ear.generate_frames(fname)
    for frame in gen:
        break
    ear.easy_imshow(frame, num=1)
