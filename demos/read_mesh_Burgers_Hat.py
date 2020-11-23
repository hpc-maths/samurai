import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from matplotlib.animation import FuncAnimation
from matplotlib import rc 
import sys


rc('text', usetex=True)
rc('font', family='serif')

filename = sys.argv[1]

f = h5py.File(filename, 'r')

# u     = f['mesh']['fields']['phi']
u     = f['mesh']['fields']['u']
level = f['mesh']['fields']['level']

points = f['mesh']['points']
connectivity = f['mesh']['connectivity']

ax1=plt.subplot(1, 2, 1)
lines = np.empty((level.size, 2, 2))
lines[:, :, 0] = points[:][connectivity[:]][:, :, 0]
lines[:, :, 1] = level[:][:, np.newaxis]

lc = mc.LineCollection(lines, colors='b', linewidths=2)
ax1.add_collection(lc)
ax1.autoscale()
ax1.set_title('level')
# ax1.set_xlim([-3, 3])
# ax1.set_ylim([0, 8])


ax2=plt.subplot(122)
sol = np.empty((u.size, 2, 2))
sol[:, :, 0] = points[:][connectivity[:]][:, :, 0]
sol[:, :, 1] = u[:][:, np.newaxis]

lc2 = mc.LineCollection(sol, colors='b', linewidths=2)
ax2.add_collection(lc2)
ax2.autoscale()

plt.title('solution')
plt.show()
