"""Plot the mesh statistics written by samurai::statistics.

samurai records mesh statistics when it is configured with the CMake option
WITH_STATS (which needs nlohmann_json). Each call to
samurai::statistics(test_case, mesh) appends the cell and interval counts of
every level of the mesh to the entry test_case of stats.json. The file is
written every 10 calls and when the program exits.

This script reads that file and draws, for one test case, the number of cells
and intervals per level along the iterations, and boxplots of the interval
sizes per level. It handles 1D, 2D and 3D meshes and levels that only appear
at some iterations.

    python read_stats.py stats.json --test-case my_case --output my_case.png

Run it from the directory that holds stats.json: the statistics file and the
output image must both lie inside the current directory, and the default
output name is the test case with every character outside [A-Za-z0-9._-]
replaced by an underscore.
"""

import argparse
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd

AXIS_NAMES = "xyz"


def axis_name(d):
    return AXIS_NAMES[d] if d < len(AXIS_NAMES) else str(d)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the mesh statistics written by a samurai build "
        "configured with -DWITH_STATS=ON."
    )
    parser.add_argument(
        "stats",
        nargs="?",
        default="stats.json",
        help="statistics file, inside the current directory (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--test-case",
        help="test case to plot, as passed to samurai::statistics "
        "(default: the only one in the file)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output image, inside the current directory (default: <test case>.png)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="output resolution (default: %(default)s)"
    )
    parser.add_argument(
        "--show", action="store_true", help="also open the figure in a window"
    )
    return parser.parse_args()


def inside_cwd(path, what):
    """Return the resolved path, or exit if it lies outside the current directory.

    Symbolic links are resolved first, so a link cannot point the script at a
    file elsewhere.
    """
    base = os.path.realpath(os.getcwd())
    resolved = os.path.realpath(path)
    if os.path.commonpath([base, resolved]) != base:
        sys.exit(
            f"{what} {path} is outside the current directory {base}, "
            "run the script from a directory that contains it"
        )
    return resolved


def default_output(test_case):
    """Build a file name in the current directory from the test case name.

    The name comes from stats.json, so it is not trusted: path separators and
    any other character outside [A-Za-z0-9._-] become underscores.
    """
    name = re.sub(r"[^A-Za-z0-9._-]", "_", test_case)
    if name in ("", ".", ".."):
        sys.exit(
            f"cannot build a file name from the test case '{test_case}', "
            "choose one with --output"
        )
    return f"{name}.png"


def select_test_case(stats, test_case, filename):
    if not stats:
        sys.exit(f"{filename} holds no test case")
    if test_case is None:
        if len(stats) > 1:
            sys.exit(
                f"{filename} holds several test cases, choose one with "
                f"--test-case: {', '.join(sorted(stats))}"
            )
        test_case = next(iter(stats))
    elif test_case not in stats:
        sys.exit(
            f"no test case '{test_case}' in {filename}, "
            f"available: {', '.join(sorted(stats))}"
        )
    return test_case


def main():
    args = parse_args()

    stats_file = inside_cwd(args.stats, "statistics file")
    with open(stats_file) as f:
        stats = json.load(f)
    test_case = select_test_case(stats, args.test_case, args.stats)
    output = inside_cwd(args.output or default_output(test_case), "output image")

    data = pd.json_normalize(stats[test_case])

    # by_level only holds the levels between the min and max level of the
    # mesh at that iteration, so a level can be missing from some rows.
    levels = sorted(
        {int(m[1]) for c in data.columns if (m := re.match(r"by_level\.(\d+)\.", c))}
    )
    dim = 1 + max(
        int(m[1]) for c in data.columns if (m := re.search(r"\.axis-(\d+)\.", c))
    )

    def plot(suffix, title, xlabel, ylabel, ax, kind="box", legend=None, stacked=True):
        fields = [f"by_level.{level:02}.{suffix}" for level in levels]
        to_plot = data.reindex(columns=fields)
        to_plot.columns = levels
        if kind == "area":
            # A level absent at an iteration has no cells nor intervals.
            to_plot = to_plot.fillna(0)
        to_plot.plot(kind=kind, ax=ax, stacked=stacked)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        if legend:
            ax.legend(title=legend)

    # The left half holds the evolution along the iterations (cells, then
    # intervals per axis), the right half the boxplots per level (cells per
    # interval for every axis, then intervals per component from the y-axis),
    # min and max side by side. In 1D the two boxplots are stacked instead.
    nrow = max(1 + dim, 2 * dim - 1)
    ncol = 4 if dim > 1 else 3

    def box_slot(row, col):
        return grid[col, 2] if dim == 1 else grid[row, 2 + col]

    fig = plt.figure(figsize=(5 * ncol, 5 * nrow))
    grid = fig.add_gridspec(nrow, ncol)

    plot(
        "cells",
        "Number of cells per level",
        "Time iteration",
        "number of cells",
        fig.add_subplot(grid[0, :2]),
        kind="area",
        legend="Level",
    )

    for d in range(dim):
        x = axis_name(d)
        plot(
            f"axis-{d}.number of intervals",
            f"Number of intervals per level in {x}-axis",
            "Time iteration",
            "number of intervals",
            fig.add_subplot(grid[1 + d, :2]),
            kind="area",
            legend="Level",
            stacked=False,
        )
        for col, (key, name) in enumerate((("min", "Minimum"), ("max", "Maximum"))):
            plot(
                f"axis-{d}.cells per interval.{key}",
                f"{name} of cells per interval\n in {x}-axis",
                "level",
                "number of cells",
                fig.add_subplot(box_slot(d, col)),
            )

    for d in range(1, dim):
        for col, (key, name) in enumerate((("min", "Minimum"), ("max", "Maximum"))):
            plot(
                f"axis-{d}.number of intervals per component.{key}",
                f"{name} of intervals in {axis_name(d - 1)}-axis\n"
                f" per {axis_name(d)}-component",
                "level",
                "number of intervals",
                fig.add_subplot(box_slot(dim + d - 1, col)),
            )

    plt.subplots_adjust(
        top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35
    )

    plt.savefig(output, dpi=args.dpi)
    print(f"{test_case}: {len(data)} iterations, levels {levels[0]} to {levels[-1]}")
    print(f"figure saved to {os.path.relpath(output)}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
