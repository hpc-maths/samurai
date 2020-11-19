import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from matplotlib.animation import FuncAnimation
from matplotlib import rc 

import sys


rc('text', usetex=True)
rc('font', family='serif')


colors = ['#000000', '#7a7a7a', '#a8a8a8']
lw = 3


def set_size(width, myratio, fraction=1):
    """ Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * myratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


width = 452.9679

fig1 = plt.figure(figsize=set_size(width, 0.35))
# ax1 = plt.subplot(1, 3, 1)
# ax2 = plt.subplot(1, 3, 2)
# ax3 = plt.subplot(1, 3, 3)

ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)


filename = sys.argv[1]
# filename = 'advection.h5'
# filename = 'advection_coarsening.h5'

f = h5py.File(filename, 'r')


u = f['fields']['u_0']

# rho = f['fields']['rho_0']
# q = f['fields']['q_0']
# E = f['fields']['e_0']


# h = f['fields']['h_0']
# q = f['fields']['q_0']


level = np.repeat(f['fields']['level_0'], 2)
mesh = f['mesh']['points']

# lines = np.empty((level.size, 2))
# lines[:, 0] = mesh[:, 0]
# lines[:, 1] = level
# lines.shape = (lines.shape[0]//2, 2, 2)
# lc = mc.LineCollection(lines, colors=colors[0], linewidths=lw)
# ax1.add_collection(lc)
# ax1.scatter(mesh[:, 0], level, marker='+', color=colors[0])
# ax1.set_xlabel("$x$")
# ax1.set_ylabel("Level $(j)$")

lines = np.empty((level.size, 2))
lines[:, 0] = mesh[:, 0]
lines[:, 1] = level
lines.shape = (lines.shape[0]//2, 2, 2)
lc = mc.LineCollection(lines, color='#95319e', linewidths=lw)
ax1.add_collection(lc)
ax1.scatter(mesh[:, 0], level, marker='+', color='#95319e')
ax1.set_xlabel("$x$")
ax1.set_ylabel("Level $(j)$")

ax1.autoscale()



# ax2.scatter(.5*(mesh[::2, 0] + mesh[1::2, 0]), u, s=lw, color=colors[0])
ax2.scatter(.5*(mesh[::2, 0] + mesh[1::2, 0]), u, s=lw, color='#95319e')

ax2.set_xlabel("$x$")



# ax2.scatter(.5*(mesh[::2, 0] + mesh[1::2, 0]), rho, s=lw, color=colors[0], label = "$\\rho$")
# ax2.scatter(.5*(mesh[::2, 0] + mesh[1::2, 0]), q, s=lw, color=colors[1], label = "$\\rho u$")
# ax2.scatter(.5*(mesh[::2, 0] + mesh[1::2, 0]), E, s=lw, color=colors[2], label = "$E$")

# ax2.scatter(.5*(mesh[::2, 0] + mesh[1::2, 0]), h, s=lw, color=colors[0], label = "$h$")
# ax2.scatter(.5*(mesh[::2, 0] + mesh[1::2, 0]), q, s=lw, color=colors[1], label = "$h u$")
# ax2.legend(fontsize = 6, ncol = 1)

# ax2.set_xlabel("$x$")

# ax2.set_ylabel("$m^{h, n}$")

# ax2.set_ylabel("$m^{0, n}$")

ax2.set_ylabel("$m^{0}(t=0, x)$")

plt.tight_layout(pad=0.4, w_pad=0.3, h_pad=1.0)

plt.show()
