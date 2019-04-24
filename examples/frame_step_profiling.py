#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

fig = plt.figure(1)
fig.clf()
ax = fig.gca()
ax.plot(frame_step, t, c='k')
ax.scatter(frame_step, t, marker='x', c='k')
ax.grid(True)
ax.set_xlabel('N')
ax.set_ylabel('runtime (s)')
ax.set_title('# frame interpolates vs. runtime')
