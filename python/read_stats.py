import pandas as pd
import matplotlib.pyplot as plt

test_case = "D2Q4444_Euler_Lax_Liu"

data = pd.read_json('stats.json')
data = pd.json_normalize(data[test_case])

min_level = data.min_level.min()
max_level = data.max_level.max()
levels = range(min_level, max_level+1)

def plot(suffix, title, xlabel, ylabel, ax, kind='box', legend=None, stacked=True):
    fields = [f'by_level.{l:02}.{suffix}' for l in levels]
    new_name = {f: l for f, l in zip(fields, levels)}

    to_plot = data[fields].rename(columns=new_name)
    to_plot.plot(kind=kind, ax=ax, stacked=stacked)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    if legend:
        ax.legend(title=legend)

nrow, ncol = 3, 4
fig = plt.figure(figsize=(5*ncol, 5*nrow))

plot("cells", "Number of cells per level", "Time iteration", "number of cells", fig.add_subplot(nrow, 2, 1), kind='area', legend="Level")

plot("axis-0.number of intervals", "Number of intervals per level in x-axis", "Time iteration", "number of intervals", fig.add_subplot(nrow, 2, 3), kind='area', legend="Level", stacked=False)
plot("axis-1.number of intervals", "Number of intervals per level in y-axis", "Time iteration", "number of intervals", fig.add_subplot(nrow, 2, 5), kind='area', legend="Level", stacked=False)

plot("axis-0.cells per interval.min", "Minimum of cells per interval\n in x-axis", "level", "number of cells", fig.add_subplot(nrow, ncol, 3))
plot("axis-0.cells per interval.max", "Maximum of cells per interval\n in x-axis", "level", "number of cells", fig.add_subplot(nrow, ncol, 4))

plot("axis-1.cells per interval.min", "Minimum of cells per interval\n in y-axis", "level", "number of cells", fig.add_subplot(nrow, ncol, 7))
plot("axis-1.cells per interval.max", "Maximum of cells per interval\n in y-axis", "level", "number of cells", fig.add_subplot(nrow, ncol, 8))

plot("axis-1.number of intervals per component.min", "Minimum of intervals in x-axis\np er y-component", "level", "number of intervals", fig.add_subplot(nrow, ncol, 11))
plot("axis-1.number of intervals per component.max", "Maximum of intervals in x-axis\n per y-component", "level", "number of intervals", fig.add_subplot(nrow, ncol, 12))

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)

# plt.show()
plt.savefig(f"{test_case}.png", dpi=300)
