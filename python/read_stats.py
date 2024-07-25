
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
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# arguments parser
CLI = argparse.ArgumentParser()

# list of PDF files
CLI.add_argument("--file", type=str,  default=[""], help="stats file")

# parse the command line
args = CLI.parse_args()

test_case = "stats"

data = pd.read_json( args.file )
data = pd.json_normalize(data[test_case])

print(type(print(data.min_level)))
print(data.min_level)

min_level = data.min_level.min()
max_level = data.max_level.max()
levels = range(min_level, max_level+1)

print("> Levels: {}".format(levels))

def plot_minmax( suffix, title, xlabel, ylabel, ax, level_range, kind='box', legend=None):
    """
        level_range (in) : range of levels (min_level, max_level)
    """
    print("\t> [plot_minmax] Entering .. ")
    # keys in dict. of data
    _tmp_fields = []
    _tmp_name   = []
    _minmaxave = {}
    for elem in ["min", "max", "ave"]:
        _fields = [f'by_level.{l:02}.{suffix}.{elem}' for l in level_range]
        
        _minmaxave[elem] =  data[ _fields ].to_numpy()[0]

        _tmp_fields.append( _fields )
        _tmp_name.append( {f: l for f, l in zip(_fields, levels)} )
    
    # ax.set_xlabel(xlabel)
    # ax.set_ylabel(ylabel)
    ax.set_xlim( level_range.start-1, level_range.stop )
    ax.set_title(title, fontweight="bold", fontsize=10)

    x = numpy.arange(min_level, max_level+1)
    spacing = 0.3  # spacing between hat groups
    width = (1 - spacing) / x.shape[0]
    ax.set_xticks( x, labels=x )
    
    ax.errorbar( x, _minmaxave["ave"], yerr=[_minmaxave["ave"]-_minmaxave["min"], _minmaxave["max"]-_minmaxave["ave"]], \
                 ls='None', marker='o', ms=6, capsize=5, c="b" )
    
    # heights0 = values[0]
    # for i, (heights, group_label) in enumerate(zip(values, group_labels)):
    #     style = {'fill': False} if i == 0 else {'edgecolor': 'black'}
    #     rects = ax.bar(x - spacing/2 + i * width, heights - heights0,
    #                    width, bottom=heights0, label=group_label, **style)
    #     label_bars(heights, rects)

    # print(to_plot)
    # print(type(to_plot))

def plot(suffix, title, xlabel, ylabel, ax, kind='box', legend=None, stacked=True):
    fields = [f'by_level.{l:02}.{suffix}' for l in levels]
    new_name = {f: l for f, l in zip(fields, levels)}

    to_plot = data[fields].rename(columns=new_name)
    to_plot.plot(kind=kind, ax=ax, stacked=stacked)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=10)
    if legend:
        ax.legend(title=legend)

nrow, ncol = 3, 4
fig = plt.figure(figsize=(5*ncol, 5*nrow))

plot("cells", "Number of cells per level", "Time iteration", "number of cells", fig.add_subplot(nrow, 2, 1), kind="box", legend="Level")

plot("axis-0.number of intervals", "Number of intervals per level in x-axis", "Time iteration", \
     "number of intervals", fig.add_subplot(nrow, 2, 3), kind='box', legend="Level", stacked=False)

plot("axis-1.number of intervals", "Number of intervals per level in y-axis", "Time iteration", \
     "number of intervals", fig.add_subplot(nrow, 2, 5), kind='box', legend="Level", stacked=False)

# plot min/max numbers of cells per level
# plot("axis-0.cells per interval.min", "Minimum of cells per interval\n in x-axis", "level", \
#      "number of cells", fig.add_subplot(nrow, ncol, 3))

# plot("axis-0.cells per interval.max", "Maximum of cells per interval\n in x-axis", "level", \
#      "number of cells", fig.add_subplot(nrow, ncol, 4))

ax = fig.add_subplot(nrow, ncol, 3)
plot_minmax("axis-0.cells per interval", "[Min, Max, Ave] Cells per interval in x-axis", "Level", \
            "number of cells", fig.add_subplot(nrow, ncol, 3), level_range=levels, kind='box', legend="Level")
plot_minmax("axis-1.cells per interval", "[Min, Max, Ave] Cells per interval in y-axis", "Level", \
            "number of cells", fig.add_subplot(nrow, ncol, 4), level_range=levels, kind='box', legend="Level")
# ax.bar(numpy.arange(min_level, max_level+1), valeurs_moyennes, yerr=[(v_min, v_max) for v_min, v_max in zip(valeurs_min, valeurs_max)], capsize=5)


plot("axis-1.cells per interval.min", "Minimum of cells per interval\n in y-axis", "level", "number of cells", fig.add_subplot(nrow, ncol, 7))
plot("axis-1.cells per interval.max", "Maximum of cells per interval\n in y-axis", "level", "number of cells", fig.add_subplot(nrow, ncol, 8))

plot("axis-1.number of intervals per component.min", "Minimum of intervals in x-axis\np er y-component", "level", "number of intervals", fig.add_subplot(nrow, ncol, 11))
plot("axis-1.number of intervals per component.max", "Maximum of intervals in x-axis\n per y-component", "level", "number of intervals", fig.add_subplot(nrow, ncol, 12))

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)

# plt.show()
plt.savefig(f"{test_case}.png", dpi=300)
