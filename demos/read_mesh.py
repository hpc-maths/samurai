import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from matplotlib.animation import FuncAnimation

import sys

filename = sys.argv[1]
# filename = 'advection.h5'
# filename = 'advection_coarsening.h5'

f = h5py.File(filename, 'r')
u = f['fields']['u_0']
# u = f['fields']['f_0']
# tag = f['fields']['tag_0']

level = np.repeat(f['fields']['level_0'], 2)
mesh = f['mesh']['points']

ax1=plt.subplot(1, 2, 1)
lines = np.empty((level.size, 2))
lines[:, 0] = mesh[:, 0]
lines[:, 1] = level
lines.shape = (lines.shape[0]//2, 2, 2)
lc = mc.LineCollection(lines, colors='b', linewidths=2)
ax1.add_collection(lc)
ax1.scatter(mesh[:, 0], level, marker='+')
ax1.autoscale()
ax1.set_title('level')
plt.subplot(122)
plt.scatter(.5*(mesh[::2, 0] + mesh[1::2, 0]), u, s=0.5)
# plt.scatter(.5*(mesh[::2, 0] + mesh[1::2, 0]), tag, s=0.5)

plt.title('solution')
plt.show()
