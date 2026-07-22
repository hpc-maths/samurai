#!/usr/bin/env python3
# Copyright 2018-2025 the samurai's authors
# SPDX-License-Identifier:  BSD-3-Clause

"""Render the 2D ghost-update cases dumped by mpi-ghost-cases to PNG.

The C++ tool mpi-ghost-cases writes one HDF5/XDMF file per case (see
dump_ghost_cases.cpp). This script rasterises the 2D ones so the meshes and
their MPI decompositions can be validated at a glance, without ParaView.

By default it builds one overview contact sheet per suite: geometry (rows) x
decomposition (cols), coloured by MPI rank, at a single stencil size (the cell
set does not depend on the stencil, only the ghost width does, which is not
drawn here). With --per-case it additionally emits, for every 2D case, a
two-panel PNG: left = cells coloured by the owning MPI rank (the decomposition),
right = cells coloured by refinement level (the geometry / level jumps).

Usage:
    python render.py --input ghost_cases_output              # overviews only
    python render.py --input ghost_cases_output --per-case   # + one PNG per case
"""

import argparse
import glob
import os
import re

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm, ListedColormap


def load_case(path):
    """Return (polys (N,4,2), rank (N,), level (N,), u (N,)) merged over ranks."""
    polys, rank, level, u = [], [], [], []
    with h5py.File(path, "r") as h:
        mesh = h["mesh"]
        for name in mesh:
            grp = mesh[name]
            pts = grp["points"][:, :2]
            conn = grp["connectivity"][()]
            polys.append(pts[conn])
            rank.append(grp["fields/rank"][()])
            level.append(grp["fields/level"][()])
            u.append(grp["fields/u"][()])
    return (
        np.concatenate(polys),
        np.concatenate(rank),
        np.concatenate(level),
        np.concatenate(u),
    )


def rank_palette(nranks):
    """Qualitative colours indexed directly by the rank value (rank k -> colour k)."""
    tab = plt.get_cmap("tab10")
    n = max(nranks, 1)
    return [tab(i % tab.N) for i in range(n)], list(range(n))


def level_palette(levels):
    """Sequential colours, one per distinct refinement level present."""
    vir = plt.get_cmap("viridis")
    n = len(levels)
    return [vir(i / max(n - 1, 1)) for i in range(n)], list(levels)


def draw_categorical(ax, polys, indices, labels, colors, title):
    """indices: per-cell colour index in [0, len(colors)); labels: tick text."""
    n = len(colors)
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(n + 1) - 0.5, n)
    pc = PolyCollection(polys, cmap=cmap, norm=norm, edgecolors="k", linewidths=0.06)
    pc.set_array(np.asarray(indices, dtype=float))
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8)
    cb = ax.figure.colorbar(pc, ax=ax, fraction=0.046, pad=0.02, ticks=range(n))
    cb.ax.set_yticklabels([str(x) for x in labels])
    cb.ax.tick_params(labelsize=6)
    return pc


def case_figure(path, nranks, name):
    """Two-panel figure (rank | level) for a single case. Returns the Figure."""
    polys, rank, level, _ = load_case(path)

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9, 4.6))

    rcolors, rlabels = rank_palette(nranks)
    draw_categorical(axl, polys, rank, rlabels, rcolors, "rank (decomposition)")

    levels = list(np.unique(level))
    lcolors, llabels = level_palette(levels)
    pos = {lv: i for i, lv in enumerate(levels)}
    lidx = [pos[v] for v in level]
    draw_categorical(axr, polys, lidx, llabels, lcolors, "level (geometry)")

    fig.suptitle(name, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# label layout: <suite>_2d_s<stencil>_<geom>[_<domain>_<periodic>]_<decomp>
GEOMS = ["corner", "patches", "adapted"]
DECOMPS = ["vstrips", "finecheck", "coarsecheck", "diagbands", "hilbert", "randomhash"]


def parse_label(name):
    m = re.match(r"(A|B)_2d_s(\d+)_(.+?)_np\d+$", name)
    if not m:
        return None
    suite, stencil, rest = m.group(1), int(m.group(2)), m.group(3)
    parts = rest.split("_")
    geom = parts[0]
    decomp = parts[-1]
    return {"suite": suite, "stencil": stencil, "geom": geom, "decomp": decomp, "rest": rest}


def overview_figure(files, suite, nranks):
    """Contact sheet: geometry (rows) x decomposition (cols), coloured by rank,
    at a single stencil size. Returns the Figure, or None if the suite is empty."""
    info = {os.path.splitext(os.path.basename(f))[0]: f for f in files}
    parsed = {n: parse_label(n) for n in info}
    parsed = {n: p for n, p in parsed.items() if p and p["suite"] == suite}
    if not parsed:
        return None

    stencil = sorted({p["stencil"] for p in parsed.values()})[0]
    geoms = [g for g in GEOMS if any(p["geom"] == g for p in parsed.values())]
    decomps = [d for d in DECOMPS if any(p["decomp"] == d for p in parsed.values())]

    rcolors, _ = rank_palette(nranks)
    cmap = ListedColormap(rcolors)
    norm = BoundaryNorm(np.arange(len(rcolors) + 1) - 0.5, len(rcolors))

    fig, axes = plt.subplots(
        len(geoms), len(decomps), figsize=(2.5 * len(decomps), 2.5 * len(geoms)), squeeze=False
    )
    for r, g in enumerate(geoms):
        for c, d in enumerate(decomps):
            ax = axes[r][c]
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            # pick the case with this geom+decomp+stencil (first match: periodic
            # variant in suite B is representative of the decomposition)
            match = [
                n for n, p in parsed.items() if p["geom"] == g and p["decomp"] == d and p["stencil"] == stencil
            ]
            if not match:
                ax.text(0.5, 0.5, "-", ha="center", va="center", transform=ax.transAxes)
            else:
                polys, rank, _, _ = load_case(info[sorted(match)[0]])
                pc = PolyCollection(polys, cmap=cmap, norm=norm, edgecolors="k", linewidths=0.04)
                pc.set_array(rank.astype(float))
                ax.add_collection(pc)
                ax.autoscale_view()
            if r == 0:
                ax.set_title(d, fontsize=9)
            if c == 0:
                ax.set_ylabel(g, fontsize=9)
    fig.suptitle(f"Suite {suite}: decomposition by rank (s{stencil})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="ghost_cases_output", help="directory holding the .h5 files")
    ap.add_argument("--output", default=None, help="output directory for the PNGs (default: <input>/png)")
    ap.add_argument(
        "--per-case",
        action="store_true",
        help="also render one PNG per individual case (rank | level); by default "
        "only the two suite overview contact sheets are produced",
    )
    ap.add_argument(
        "--all-stencils",
        action="store_true",
        help="with --per-case, render every stencil size; by default only one is "
        "kept per case (the drawn cells and decomposition do not depend on the "
        "stencil, only the ghost width does, which is not drawn)",
    )
    args = ap.parse_args()

    out_dir = args.output or os.path.join(args.input, "png")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(
        f
        for f in glob.glob(os.path.join(args.input, "*_2d_*.h5"))
        if os.path.basename(f).startswith(("A_2d_", "B_2d_"))
    )
    if not files:
        raise SystemExit(f"no 2D case files (A_2d_*.h5 / B_2d_*.h5) found in {args.input}")

    # infer the number of ranks from the file name suffix _np<K>
    nranks = 1
    for f in files:
        m = re.search(r"_np(\d+)\.h5$", os.path.basename(f))
        if m:
            nranks = max(nranks, int(m.group(1)))

    ncases = 0
    if args.per_case:
        # The drawn cells / rank / level do not depend on the stencil size, so by
        # default keep a single representative per case (smallest stencil) and
        # drop the s<n> tag from its name; --all-stencils renders them all.
        to_render = []  # (path, output_name)
        if args.all_stencils:
            for f in files:
                to_render.append((f, os.path.splitext(os.path.basename(f))[0]))
        else:
            groups = {}  # (suite, rest) -> (stencil, path)
            for f in files:
                p = parse_label(os.path.splitext(os.path.basename(f))[0])
                if p is None:
                    continue
                key = (p["suite"], p["rest"])
                if key not in groups or p["stencil"] < groups[key][0]:
                    groups[key] = (p["stencil"], f)
            for (suite, rest), (_, f) in sorted(groups.items()):
                to_render.append((f, f"{suite}_2d_{rest}"))

        for f, name in to_render:
            fig = case_figure(f, nranks, name)
            fig.savefig(os.path.join(out_dir, name + ".png"), dpi=130)
            plt.close(fig)
            print("  " + name)
        ncases = len(to_render)

    for suite in ("A", "B"):
        fig = overview_figure(files, suite, nranks)
        if fig is not None:
            fig.savefig(os.path.join(out_dir, f"overview_suite_{suite}.png"), dpi=140)
            plt.close(fig)
            print(f"  overview_suite_{suite}")

    print("Rendered 2 overview(s)" + (f" + {ncases} case(s)" if ncases else "") + f" to {out_dir}")


if __name__ == "__main__":
    main()
