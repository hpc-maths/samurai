
#
#
# categories = ['A', 'B', 'C', 'D']
# valeurs_moyennes = [3, 7, 5, 9]
# valeurs_min = [2, 5, 4, 8]
# valeurs_max = [4, 9, 6, 10]

# # Création du graphique
# plt.figure(figsize=(10, 6))

# # Tracé des barres
# plt.bar(categories, valeurs_moyennes, yerr=[(v_min, v_max) for v_min, v_max in zip(valeurs_min, valeurs_max)], capsize=5)

import numpy
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_mma( data, suffix, ax, xlabel, ylabel, keys=None ):
    """
    """
    
    # select specific domaine if not None
    if keys == None:
        keys = data.keys()
    
    print("plot data for domaine {}".format(keys))

    for mprank in keys:
        levels = range( data[ mprank ].min_level.min(), data[ mprank ].max_level.max()+1 )

        for elem in ["min", "max", "ave"]:
            fields = [f'by_level.{l:02}.{suffix}.{elem}' for l in levels]
            print("\t>[plot] Getting field: '{}'".format(fields))

    #     x = numpy.arange( levels.start, levels.stop )
    #     y = data[ mprank ][ fields ].to_numpy()[0]

    #     ax.plot( x, y, ls='None', markersize=5, marker='o' )
    
    ax.set_xlabel( xlabel, fontsize=10 )
    ax.set_ylabel( ylabel, fontsize=10 )

def plot_by_level( data, suffix, ax, xlabel, ylabel, keys=None ):
    """
        Plot data level by level
    """
    
    # select specific domaine if not None
    if keys == None:
        keys = data.keys()
    
    print("plot data for domaine {}".format(keys))

    for mprank in keys:
        levels = range( data[ mprank ].min_level.min(), data[ mprank ].max_level.max()+1 )
        fields = [f'by_level.{l:02}.{suffix}' for l in levels]

        x = numpy.arange( levels.start, levels.stop )
        y = data[ mprank ][ fields ].to_numpy()[0]

        ax.plot( x, y, ls='None', markersize=5, marker='o' )
    
    ax.set_xlabel( xlabel, fontsize=10 )
    ax.set_ylabel( ylabel, fontsize=10 )


def plot_single_quantity( data, suffix, ax, xlabel, ylabel, keys=None ):
    """
        Plot data level by level
    """
    
    # select specific domaine if not None
    if keys == None:
        keys = list(data.keys())

    rmin = int( min(keys) )
    rmax = int( max(keys) )

    x = numpy.zeros( len(keys) )
    y = numpy.zeros( x.shape )

    for id in range(0, len(keys)):
        mprank = int( keys[ id ] )
        x[ id ] = mprank
        y[ id ] = data[ str(mprank) ][suffix].values[0]
        print("\t>[plot] Process # {}, data single: '{}'".format( mprank, data[ str(mprank)][suffix].values ))

    ax.plot( x, y, ls='None', markersize=5, marker='o' )
    
    ax.set_xlabel( xlabel, fontsize=10 )
    ax.set_ylabel( ylabel, fontsize=10 )

# arguments parser
CLI = argparse.ArgumentParser()
CLI.add_argument("--files", nargs="*", type=str,  default=[""], help="stats file")
CLI.add_argument("--jtag", type=str, default="stats", help="json tag")

# parse the command line
args = CLI.parse_args()

min_level = 99
max_level = 0
process_stats = {}
for fin in args.files:

    print("\t> Parsing file: '{}'".format( fin ))
    fdata = pd.read_json( fin )
    fdata = pd.json_normalize( fdata[ args.jtag ] )

    match = re.search(r'process_(\d+)', fin)
    if match:
        rank = match.group(1)
    else:
        rank = 0
    
    # compute min/max level overall datasets
    min_level = min( min_level, fdata.min_level.min() )
    max_level = max( max_level, fdata.max_level.max() )

    assert rank not in process_stats.keys(), "statistics for process {} exists".format(rank)
    process_stats[ rank ] = fdata


nrow, ncol = 3, 2
fig = plt.figure(figsize=(5*ncol, 5*nrow))

# plot("cells", "Number of cells per level", "Level", "number of cells",  )

ax = fig.add_subplot(nrow, ncol, 1)
plot_by_level( process_stats, "cells", ax, xlabel="level", ylabel="Number of cells" )
ax.set_xlim( min_level-1, max_level+1 )
ax.set_title( "Number of cells", fontweight="bold", fontsize=10 )
x = numpy.arange(min_level, max_level+1)
ax.set_xticks( x, labels=x )

ax = fig.add_subplot(nrow, ncol, 2)
plot_by_level( process_stats, "axis-0.number of intervals", ax, xlabel="level", ylabel="Number of intervals" )
ax.set_xlim( min_level-1, max_level+1 )
ax.set_title( "Number of intervals", fontweight="bold", fontsize=10 )
x = numpy.arange(min_level, max_level+1)
ax.set_xticks( x, labels=x )

ax = fig.add_subplot(nrow, ncol, 3)
plot_single_quantity( process_stats, "n_neighbours", ax, xlabel="MPI rank", ylabel="Number of neighbours" )
ax.set_title( "Number of geometrical neighbours", fontweight="bold", fontsize=10 )
# x = numpy.arange(min_level, max_level+1)
# ax.set_xticks( x, labels=x )

ax = fig.add_subplot(nrow, ncol, 4)
plot_mma( process_stats, "axis-0.cells per interval", ax, xlabel="level", ylabel="Number of cells" )
ax.set_xlim( min_level-1, max_level+1 )
ax.set_title( "Number of cells per interval x-axis", fontweight="bold", fontsize=10 )
x = numpy.arange(min_level, max_level+1)
ax.set_xticks( x, labels=x )


plt.savefig( f"{args.jtag}.png", dpi=300 )
