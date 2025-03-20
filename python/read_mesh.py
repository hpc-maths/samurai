# Copyright 2021 SAMURAI TEAM. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib import animation
from matplotlib import rc
import argparse

def read_mesh(filename, ite=None):
    return h5py.File(filename + '.h5', 'r')['mesh']

def scatter_plot(ax, points):
    return ax.scatter(points[:, 0], points[:, 1], marker='+')

def scatter_update(scatter, points):
    scatter.set_offsets(points[:, :2])

def line_plot(ax, x, y):
    return ax.plot(x, y, 'C0o-', linewidth=1, markersize=3, alpha=0.5)[0]

def line_update(line, x, y):
    line.set_data(x, y)

class Plot:
    def __init__(self, filename):
        self.fig = plt.figure()
        self.artists = []
        self.ax = []

        if args.field is None:
            ax = plt.subplot(111)
            mesh = read_mesh(filename)
            if 'points' in mesh:
                self.plot(ax, mesh)
            else:
                for rank in mesh.keys():
                    self.plot(ax, mesh[rank])
            ax.set_title("Mesh")
            self.ax = [ax]
        else:
            for i, f in enumerate(args.field):
                ax = plt.subplot(1, len(args.field), i + 1)
                mesh = read_mesh(filename)

                if 'points' in mesh:
                    unknown_field = next((f for f in args.field if f not in mesh['fields']), None)

                    if unknown_field is not None:
                        keys = ' '.join(mesh['fields'].keys())
                        raise ValueError(f"file:{filename}> field:{unknown_field} not in available fields values: {keys}")

                    self.plot(ax, mesh, f)
                else:
                    for rank in mesh.keys():
                        unknown_field = next((f for f in args.field if f not in mesh[rank]['fields']), None)

                        if unknown_field is not None:
                            keys = ' '.join(mesh[rank]['fields'].keys())
                            raise ValueError(f"file:{filename}> field:{unknown_field} not in available fields values: {keys}")
                        self.plot(ax, mesh[rank], f)
                ax.set_title(f)
                self.ax.append(ax)

    def plot(self, ax, mesh, field=None, init=True):
        points = mesh['points']
        connectivity = mesh['connectivity']

        segments = np.zeros((connectivity.shape[0], 2, 2))
        segments[:, :, 0] = points[:][connectivity[:]][:, :, 0]

        if field is None:
            segments[:, :, 1] = 0
            if init:
                self.artists.append(scatter_plot(ax, points))
                self.lc = mc.LineCollection(segments, colors='b', linewidths=2)
                self.lines = ax.add_collection(self.lc)
            else:
                scatter_update(self.artists[self.index], points)
                self.index += 1
                # self.lc.set_array(segments)
        else:
            data = mesh['fields'][field][:]
            centers = .5*(segments[:, 0, 0] + segments[:, 1, 0])
            segments[:, :, 1] = data[:, np.newaxis]
            # ax.scatter(centers, data, marker='+')
            index = np.argsort(centers)
            if init:
                self.artists.append(line_plot(ax, centers[index], data[index]))
            else:
                line_update(self.artists[self.index], centers[index], data[index])
                self.index += 1

        for aax in self.ax:
            aax.relim()
            aax.autoscale_view()

    def update(self, filename):
        self.index = 0
        if args.field is None:
            if args.mpi_size == 1:
                mesh = read_mesh(filename)
                self.plot(None, mesh, init=False)
            else:
                for rank in range(args.mpi_size):
                    mesh = read_mesh(f"{filename}_rank_{rank}")
                    self.plot(None, mesh, init=False)

        else:
            for i, f in enumerate(args.field):
                if args.mpi_size == 1:
                    mesh = read_mesh(filename)
                    self.plot(None, mesh, f, init=False)
                else:
                    for rank in range(args.mpi_size):
                        mesh = read_mesh(f"{filename}_rank_{rank}")
                        self.plot(None, mesh, f, init=False)

    def get_artist(self):
        return self.artists

parser = argparse.ArgumentParser(description='Plot 1d mesh and field from samurai simulations.')
parser.add_argument('filename', type=str, help='hdf5 file to plot without .h5 extension')
parser.add_argument('--field', nargs="+", type=str, required=False, help='list of fields to plot')
parser.add_argument('--start', type=int, required=False, default=0, help='iteration start')
parser.add_argument('--end', type=int, required=False, default=None, help='iteration end')
parser.add_argument('--save', type=str, required=False, help='output file')
parser.add_argument('--mpi-size', type=int, default=1, required=False, help='number of mpi rank')
parser.add_argument('--wait', type=int, default=200, required=False, help='time between two plot in ms')
args = parser.parse_args()

if args.end is None:
    Plot(args.filename)
else:
    p = Plot(f"{args.filename}{args.start}")
    def animate(i):
        p.fig.suptitle(f"iteration {i + args.start}")
        p.update(f"{args.filename}{i + args.start}")
        return p.get_artist()
    ani = animation.FuncAnimation(p.fig, animate, frames=args.end-args.start, interval=args.wait, repeat=True)

if args.save:
    if args.end is None:
        plt.savefig(args.save + '.png', dpi=300)
    else:
        writermp4 = animation.FFMpegWriter(fps=1)
        ani.save(args.save + '.mp4', dpi=300)
else:
    plt.show()
