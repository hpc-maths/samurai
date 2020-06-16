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
plt.title('solution')
plt.show()

# fig, (ax1, ax2) = plt.subplots(1, 2)

# filename = f'advection_0.h5'

# f = h5py.File(filename, 'r')
# u = f['fields']['u']
# level = np.repeat(f['fields']['level'], 2)
# mesh = f['mesh']['points']

# lines = np.empty((level.size, 2))
# lines[:, 0] = mesh[:, 0]
# lines[:, 1] = level
# lines.shape = (lines.shape[0]//2, 2, 2)
# lc = mc.LineCollection(lines, colors='b', linewidths=2)
# collection = ax1.add_collection(lc)
# scatter_1 = ax1.scatter(mesh[:, 0], level, marker='+')
# ax1.autoscale()
# scatter_2 = ax2.scatter(.5*(mesh[::2, 0] + mesh[1::2, 0]), u)

# # for i in range(0, 1000, 10):
# #     filename = f'advection_{i}.h5'

# #     f = h5py.File(filename, 'r')
# #     u = f['fields']['u']
# #     level = np.repeat(f['fields']['level'], 2)
# #     mesh = f['mesh']['points']

# #     lines = np.empty((level.size, 2))
# #     lines[:, 0] = mesh[:, 0]
# #     lines[:, 1] = level
# #     lines.shape = (lines.shape[0]//2, 2, 2)
# #     lc = mc.LineCollection(lines, colors='b', linewidths=2)
# #     ax1.add_collection(lc)
# #     ax1.scatter(mesh[:, 0], level, marker='+')
# #     ax1.autoscale()
# #     ax2.scatter(.5*(mesh[::2, 0] + mesh[1::2, 0]), u)

# #     plt.show()

# # def animate(frame):
# #     print(frame)
# #     filename = f'advection_{frame}.h5'

# #     f = h5py.File(filename, 'r')
# #     u = f['fields']['u']
# #     level = np.repeat(f['fields']['level'], 2)
# #     mesh = f['mesh']['points']

# #     lines = np.empty((level.size, 2))
# #     lines[:, 0] = mesh[:, 0]
# #     lines[:, 1] = level
# #     lines.shape = (lines.shape[0]//2, 2, 2)
# #     # lc = mc.LineCollection(lines, colors='b', linewidths=2)

# #     scatter_1.set_offsets(mesh[:, 0])
# #     scatter_1.set_array(level)

# #     scatter_2.set_offsets(.5*(mesh[::2, 0] + mesh[1::2, 0]))
# #     print(.5*(mesh[::2, 0] + mesh[1::2, 0]))
# #     scatter_2.set_array(u)

# #     collection.set_paths(lines)

# # for frame in range(500):
# #     ax1.clear()
# #     ax2.clear()

# #     filename = f'advection_{frame}.h5'

# #     print(frame)
# #     f = h5py.File(filename, 'r')
# #     u = f['fields']['u']
# #     level = np.repeat(f['fields']['level'], 2)
# #     mesh = f['mesh']['points']

# #     lines = np.empty((level.size, 2))
# #     lines[:, 0] = mesh[:, 0]
# #     lines[:, 1] = level
# #     lines.shape = (lines.shape[0]//2, 2, 2)
# #     lc = mc.LineCollection(lines, colors='b', linewidths=2)
# #     collection = ax1.add_collection(lc)
# #     scatter_1 = ax1.scatter(mesh[:, 0], level, marker='+')
# #     ax1.autoscale()
# #     scatter_2 = ax2.scatter(.5*(mesh[::2, 0] + mesh[1::2, 0]), u)


# #     plt.show(block=False)

# #     plt.pause(0.1)
