"""The figures of ``tutorial/1d_burgers_amr.md``: the cells, the criterion and the result.

All three figures draw the final mesh of the program of the page,
``tutorial/snippet/burgers/burgers_1d.cpp`` run with its default options, and the
plate draws its solution ``u`` too. ``data/1d_burgers_amr.json`` holds both: the
cells as ``docs/tools/export_cells.py`` writes them, and ``u``, one value per cell
from left to right. Run this module to write it again from ``burgers_1d.h5``::

    python -m samurai_figures.pages.1d_burgers_amr burgers_1d.h5

from ``docs/source/_ext``, in an environment with ``h5py``.
"""

import argparse
import importlib.util
import json
from functools import lru_cache
from math import sqrt
from pathlib import Path

from ..cells import check_disjoint, check_graded, load_cells
from ..draw import (
    EMPTY,
    FIG,
    HAIR,
    HEAVY,
    INK,
    INK2,
    PAPER,
    RED,
    RULE,
    WASH,
    bracket,
    callout,
    fig_label,
    line,
    num,
    patterns,
    rect,
    text,
)
from ..plots import Frame, axes, curve, level_rows, line_key, steps
from ..registry import DIAGRAM_WIDTH, PLATE_WIDTH, Drawing, figure

DATA = Path(__file__).resolve().parents[1] / "data" / "1d_burgers_amr.json"
SOURCE = "burgers_1d.cpp (tutorial/snippet/burgers), default options"

# What the page says about the run; ``_run`` checks the data against it.
T_FINAL, A, B = 1.5, -3.0, 3.0
NB_CELLS = 115
COUNTS = {8: 100, 7: 4, 6: 3, 5: 3, 4: 3, 3: 2}
FINEST = (-1.03125, 1.3125)
# the shock as the legend of the plate gives it: the peak of u, then the three
# cells of [1.1953125, 1.265625) where u falls
PEAK, SHOCK_CELLS = 0.870, ((1.1953125, 0.865), (1.21875, 0.569), (1.2421875, 0.053))

MINUS = "−"


def _sub(s, size):
    """Return ``s`` as the subscript of a math label of font size ``size``.

    The subscript is set at 0.7 of the size and lowered by 0.2 of it. Chrome
    lowers ``baseline-shift="sub"`` by about 0.8 of the size of the serif italic,
    which leaves a subscript of several digits hanging below its letter.
    """
    return f'<tspan baseline-shift="{num(-0.2 * size)}" font-size="{num(0.7 * size)}">{s}</tspan>'


def _math(x, y, s, size=14, fill=INK, anchor="middle"):
    """Return a math label at (x, y), in the serif italic."""
    return text(x, y, s, anchor=anchor, size=size, fill=fill, cls="sm-fig-math")


class Run:
    """The final state of the run: its cells from left to right, and ``u`` on them.

    ``cells`` holds ``(level, k)``, the cell ``C_{level,k}`` of the page;
    ``u[n]`` is the value of the cell ``cells[n]``, whose flat index is ``n``.
    """

    def __init__(self, exported, u):
        """Sort the cells of ``exported`` from left to right and pair them with ``u``."""
        self.h = exported.scaling_factor
        self.a = exported.origin[0]
        self.cells = sorted(((lv, k) for lv, k in exported.leaves), key=self.left)
        self.u = u

    def left(self, cell):
        """Return the left end of ``cell``, in the coordinates of the program."""
        level, k = cell
        return self.a + k * self.h / 2**level

    def right(self, cell):
        """Return the right end of ``cell``."""
        level, k = cell
        return self.a + (k + 1) * self.h / 2**level

    def flat(self, cell):
        """Return the flat index of ``cell``, its rank from the left."""
        return self.cells.index(cell)


@lru_cache(maxsize=1)
def _run():
    """Load the run and check it: graded, covering [a, b), as the page describes it."""
    with open(DATA, encoding="utf-8") as stream:
        u = json.load(stream)["u"]
    exported = load_cells(DATA)
    run = Run(exported, u)
    # the cells as 2D cells of one row, for the checks of cells.py
    plane = [(level, k, 0) for level, k in run.cells]
    check_disjoint(plane)
    check_graded(plane)
    for left, right in zip(run.cells, run.cells[1:]):
        if abs(run.right(left) - run.left(right)) > 1e-12:
            raise ValueError(f"the cells {left} and {right} leave a gap or overlap")
    if run.left(run.cells[0]) != A or run.right(run.cells[-1]) != B:
        raise ValueError("the cells do not cover [-3, 3)")
    counts = {}
    for level, _ in run.cells:
        counts[level] = counts.get(level, 0) + 1
    if len(run.cells) != NB_CELLS or len(u) != NB_CELLS or counts != COUNTS:
        raise ValueError(f"the run has {len(run.cells)} cells, by level {counts}")
    finest = [c for c in run.cells if c[0] == max(COUNTS)]
    if (run.left(finest[0]), run.right(finest[-1])) != FINEST or len(finest) != COUNTS[8]:
        raise ValueError("the cells of level 8 are not one run on [-1.03125, 1.3125)")
    by_left = {run.left(c): v for c, v in zip(run.cells, u)}
    if round(max(u), 3) != PEAK or any(round(by_left[x], 3) != v for x, v in SHOCK_CELLS):
        raise ValueError("u around the shock is not what the legend of the plate says")
    return run


def _window(run, lo, hi):
    """Return the cells of ``run`` inside [lo, hi), from left to right."""
    return [c for c in run.cells if lo <= run.left(c) and run.right(c) <= hi]


# --- the cells and the fluxes ---------------------------------------------------------


@figure
def burgers_cells(_p):
    """Draw five cells of the final mesh, in both notations, and the flux at each face.

    Figure of tutorial/1d_burgers_amr.md. The cells are those of [1.265625, 1.5),
    right of the shock: two of level 8, two of level 7, one of level 6. Above each
    cell its flat index, counted from C_0 at x = -3 as the page does. Each face
    carries a tick and the name of its flux, with no arrow: the flux of a face has
    no fixed direction, and right of the shock u changes sign from cell to cell.
    Above the cells of level 7, set apart and dashed, their parent C_{6,46}, which
    is not a cell of the mesh.
    """
    run = _run()
    lo, hi = 1.265625, 1.5
    cells = _window(run, lo, hi)
    if [run.flat(c) for c in cells] != list(range(106, 111)):
        raise ValueError("the five cells are not C_106 to C_110 of the run")
    unit = 600 / (hi - lo)  # drawing units per unit of x
    x0, top, ch, stub = 40, 92, 42, 30

    def X(x):
        return x0 + (x - lo) * unit

    g = ""
    # the mesh goes on at both ends: open stubs of the cells beyond
    for xa, xb in ((x0 - stub, x0), (X(hi), X(hi) + stub)):
        g += line(xa, top, xb, top, stroke=FIG, sw=EMPTY, dash="3 3")
        g += line(xa, top + ch, xb, top + ch, stroke=FIG, sw=EMPTY, dash="3 3")
    # the parent of the two cells of level 7, not a cell of the mesh
    kids = [c for c in cells if c[0] == 7]
    parent = (6, kids[0][1] // 2)
    if [(7, 2 * parent[1]), (7, 2 * parent[1] + 1)] != kids or parent in run.cells:
        raise ValueError(f"{kids} are not the two children of {parent}, outside the mesh")
    pa, pb = X(run.left(kids[0])), X(run.right(kids[1]))
    g += rect(pa, 10, pb - pa, 30, stroke=INK2, dash="4 3")
    g += _math((pa + pb) / 2, 30, f"C{_sub(f'{parent[0]},{parent[1]}', 14)}", fill=INK2)
    for xx in (pa, (pa + pb) / 2, pb):
        g += line(xx, 40, xx, top - 24, stroke=INK2, sw=HAIR, dash="2 2")
    g += text(pb + 14, 22, "not a cell of the mesh:", size=11.5, fill=INK2)
    g += text(pb + 14, 37, "the parent of the two cells below", size=11.5, fill=INK2)

    faces = [X(run.left(c)) for c in cells] + [X(run.right(cells[-1]))]
    jumps = {q for q in range(1, len(cells)) if cells[q - 1][0] != cells[q][0]}
    for c in cells:
        xa, xb = X(run.left(c)), X(run.right(c))
        g += rect(xa, top, xb - xa, ch, fill=WASH, stroke=INK)
        g += _math((xa + xb) / 2, top + 26, f"C{_sub(f'{c[0]},{c[1]}', 14)}")
        g += _math((xa + xb) / 2, top - 8, f"C{_sub(run.flat(c), 13)}", size=13, fill=INK2)
    # one flux a face, named after the flat index of the cell on its left; red at a
    # level jump
    tick = top + ch + 12
    for q, xf in enumerate(faces):
        red = q in jumps
        k = run.flat(cells[0]) + q - 1
        g += line(xf, top + ch, xf, tick, stroke=RED if red else INK, sw=RULE)
        g += _math(xf, tick + 18, f"F{_sub(f'{k}+1/2', 13)}", size=13, fill=RED if red else INK)
        if red:
            g += line(xf, top - 3, xf, top + ch + 3, stroke=RED, sw=HEAVY)
    return Drawing(
        DIAGRAM_WIDTH,
        tick + 28,
        "Five neighboring cells of the final mesh of the run, right of the shock: C 8,182 and"
        " C 8,183 of level 8, C 7,92 and C 7,93 of level 7, C 6,47 of level 6. Above each cell"
        " its flat index, C 106 to C 110. Under each face a tick and the name of its flux,"
        " F 105+1/2 to F 110+1/2; the faces between levels 8 and 7 and between levels 7 and 6"
        " are red. Above the two cells of level 7, set apart and dashed, their parent C 6,46,"
        " marked as not a cell of the mesh.",
        "",
        g,
    )


# --- the shifted intervals of the criterion -------------------------------------------


@figure
def burgers_criterion(p):
    """Draw the interval of level 7 [92, 94) and the two shifted intervals the criterion reads.

    Figure of tutorial/1d_burgers_amr.md. The top row is the final mesh over the
    cells 90 to 95 of level 7; below it, ``u(level, i - 1)``, ``u(level, i)`` and
    ``u(level, i + 1)`` for the interval ``i`` = [92, 94) of that mesh. The
    neighbors at both ends, 91 and 94, are ghost cells, drawn dashed.
    """
    run = _run()
    level, first, last = 7, 90, 96
    interval = [k for lv, k in run.cells if lv == level and 92 <= k < 94]
    if interval != [92, 93] or (level, 91) in run.cells or (level, 94) in run.cells:
        raise ValueError("[92, 94) is not a whole interval of level 7 in the run")
    x0, cw, ch = 150, 72, 26

    def X(k):
        return x0 + (k - first) * cw

    lo = run.a + first * run.h / 2**level
    hi = run.a + last * run.h / 2**level
    g = ""
    for k in range(first, last):
        g += text(X(k) + cw / 2, 14, str(k), anchor="middle", size=11, fill=INK2, cls="sm-fig-code")
    # the mesh: the cells of every level over the same stretch of x
    y = 24
    for c in run.cells:
        if lo <= run.left(c) and run.right(c) <= hi:
            xa = x0 + (run.left(c) - lo) / (hi - lo) * cw * (last - first)
            xb = x0 + (run.right(c) - lo) / (hi - lo) * cw * (last - first)
            g += rect(xa, y, xb - xa, ch, stroke=INK, sw=EMPTY)
            g += text((xa + xb) / 2, y + 17, str(c[0]), anchor="middle", size=11, fill=INK2)
    g += text(x0 - 14, y + 17, "level of the cells", anchor="end", size=12, fill=INK)
    rows = (("u(level, i - 1)", -1), ("u(level, i)", 0), ("u(level, i + 1)", 1))
    for r, (name, shift) in enumerate(rows):
        y = 82 + r * 60
        for k in range(first, last):
            g += rect(X(k), y, cw, ch, stroke=FIG, sw=EMPTY)
        for k in (kk + shift for kk in interval):
            if (level, k) in run.cells:
                g += rect(X(k), y, cw, ch, fill=WASH, stroke=INK)
            else:
                g += rect(X(k), y, cw, ch, fill=PAPER, stroke=INK2, dash="4 3")
        a, b = interval[0] + shift, interval[-1] + 1 + shift
        g += bracket(X(a), X(b), y + ch + 9, f"[{a}, {b})", size=11)
        g += text(x0 - 14, y + 17, name, anchor="end", size=11.5, fill=INK, cls="sm-fig-code")
    return Drawing(
        DIAGRAM_WIDTH,
        82 + 2 * 60 + ch + 30,
        "Columns for the cells 90 to 95 of level 7. Top row, the final mesh there: two cells"
        " of level 7 between level 8 on the left and a cell of level 6 on the right. Below,"
        " three rows: u(level, i - 1) covers [91, 93), u(level, i) covers [92, 94) and"
        " u(level, i + 1) covers [93, 95). The cells 91 and 94 are ghost cells, drawn dashed.",
        patterns(p),
        g,
    )


# --- the result -------------------------------------------------------------------------


def exact(x, t=T_FINAL):
    """Return the exact solution after the shock, eq. burgers-solution-after-shock."""
    s = sqrt(2 * (1 + t)) - 1
    return (1 + x) / (1 + t) if -1 <= x <= s else 0.0


@figure
def burgers_result(p):
    """Draw u at t = 1.5 with the exact solution, and the levels of the 115 cells.

    Plate of tutorial/1d_burgers_amr.md. Fig. 1 is u, cell by cell, over the exact
    solution; Fig. 2 the cells of each level, on the same x axis.
    """
    run = _run()
    x0, w = 96, 528
    xticks = [(v, str(v).replace("-", MINUS)) for v in range(-3, 4)]
    fr = Frame(x0, 34, w, 200, xlim=(A, B), ylim=(-0.05, 1.05))
    shock = sqrt(2 * (1 + T_FINAL)) - 1
    g = axes(fr, xticks=xticks, yticks=[(0, "0"), (0.5, "0.5"), (1, "1")], xname="x", yname="u")
    g += curve(fr, exact, hi=shock)
    g += curve(fr, exact, lo=shock + 1e-9)
    g += line(fr.X(shock), fr.Y(exact(shock)), fr.X(shock), fr.Y(0), stroke=INK, sw=HEAVY)
    cells = [(run.left(c), run.right(c), v) for c, v in zip(run.cells, run.u)]
    g += steps(fr, cells, stroke=RED, sw=RULE)
    g += line_key(
        fr.X(-2.85),
        fr.Y(0.93),
        [
            ("exact solution", INK, HEAVY, ""),
            (f"u, {len(run.cells)} cells", RED, RULE, ""),
        ],
    )
    g += callout(fr.X(1.23), fr.Y(0.57), fr.X(1.85), fr.Y(0.72), "a", red=True)
    g += callout(fr.X(-1.0), fr.Y(0.0), fr.X(-1.55), fr.Y(0.32), "b")
    g += fig_label(fr.x + w / 2, fr.bottom + 50, 1, "t = 1.5")

    top = fr.bottom + 96
    leaves = [(level, k) for level, k in run.cells]
    g += level_rows(x0, top, w, leaves, p)
    nrows = max(COUNTS) - min(COUNTS) + 1
    bottom = top + (nrows - 1) * 30 + 16
    band = [c for c in run.cells if c[0] == max(COUNTS)]
    xa, xb = fr.X(run.left(band[0])), fr.X(run.right(band[-1]))
    g += bracket(xa, xb, bottom + 12, f"[{MINUS}1.03125, 1.3125)", color=RED, size=11)
    g += callout(fr.X(2.6), top + 8, fr.X(2.6), top - 22, "c")
    axis = Frame(x0, bottom + 32, w, 0, xlim=(A, B))
    g += axes(axis, xticks=xticks, xname="x")
    g += fig_label(fr.x + w / 2, axis.bottom + 50, 2)
    height = axis.bottom + 62
    split = fr.bottom + 66
    return Drawing(
        PLATE_WIDTH,
        height,
        "Two figures over x from -3 to 3. Fig. 1: u at t = 1.5, as 115 red steps, over the"
        " exact solution, a heavy line that is zero, rises as a ramp from x = -1 to the shock"
        " at x = 1.236, where it drops to zero. The steps follow the ramp and spread the shock"
        " over a few cells. Fig. 2: one row per level, from level 3 at the top to level 8: 2,"
        " 3, 3, 3, 4 and 100 cells. The 100 cells of level 8, hatched in red, cover"
        " [-1.03125, 1.3125); the levels drop towards the boundaries down to level 3.",
        patterns(p),
        g,
        panels=(
            (0, 0, PLATE_WIDTH, split, "wide"),
            (0, split, PLATE_WIDTH, height - split, "wide"),
        ),
    )


# --- writing the data -------------------------------------------------------------------


def _exporter():
    """Load the module ``docs/tools/export_cells.py``."""
    path = Path(__file__).resolve().parents[4] / "tools" / "export_cells.py"
    spec = importlib.util.spec_from_file_location("export_cells", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export(h5, out=DATA):
    """Write the cells and ``u`` of ``h5``, the file ``burgers_1d.h5``, to ``out``.

    The cells go through ``export_cells.export``; ``u`` is read from the same
    file and sorted from left to right, rounded to 6 decimals.
    """
    import h5py  # pylint: disable=import-outside-toplevel

    tools = _exporter()
    document = tools.export(h5, scaling_factor=B - A, origin=[A], source=SOURCE)
    with h5py.File(h5, "r") as f:
        points = f["mesh/points"][()]
        lefts = points[f["mesh/connectivity"][()]][:, :, 0].min(axis=1)
        values = f["mesh/fields/u"][()].tolist()
    u = [float(f"{v:.6f}") + 0.0 for _, v in sorted(zip(lefts.tolist(), values))]
    rows = ",\n".join(
        "    " + ", ".join(json.dumps(v) for v in u[q : q + 8]) for q in range(0, len(u), 8)
    )
    text_ = tools.dumps(document).rstrip().rstrip("}").rstrip()
    Path(out).write_text(f'{text_},\n  "u": [\n{rows}\n  ]\n}}\n', encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write data/1d_burgers_amr.json.")
    parser.add_argument("h5", type=Path, help="the burgers_1d.h5 file of the run")
    export(parser.parse_args().h5)
