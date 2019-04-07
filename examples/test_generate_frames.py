#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import context
import ear

if __name__ == '__main__':

    fname = r'..\data\original\video\office.mp4'
    gen = ear.generate_frames(fname)
    for frame in gen:
        break
    ear.easy_imshow(frame, num=1)
