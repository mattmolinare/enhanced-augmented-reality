#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pstats

stats_dir = r'..\output\fs'
stats_files = sorted(glob.iglob(os.path.join(stats_dir, '*.stats')))

num_frames = 1291
frame_step = [1, 2, 4, 8, 16, 32]

t = []
for file in stats_files:
    frame_step
    p = pstats.Stats(file)
    t.append(p.total_tt)
t = np.array(t)

matplotlib.rcParams.update({'font.size': 22})
fig = plt.figure(1, figsize=(8, 6))
fig.clf()
ax = fig.gca()
ax.plot(frame_step, num_frames / t, c='k', ls='--')
ax.scatter(frame_step, num_frames / t, marker='s', c='k', s=30)
ax.grid(True)
ax.set_xlabel('N')
ax.set_ylabel('FPS')
ax.set_title('')
