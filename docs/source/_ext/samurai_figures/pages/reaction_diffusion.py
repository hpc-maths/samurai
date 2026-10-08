"""The figure of ``tutorial/reaction_diffusion.md``: the traveling wave and its mesh.

The plate draws two states of the demo ``demos/FiniteVolume/nagumo.cpp``, run with
its default options: the files ``nagumo_ite_0.h5`` (t = 0, after the first mesh
adaptation) and ``nagumo_ite_100.h5`` (t = 1, after the 100th time step). Each
state is a file of ``data/``: the cells as ``docs/tools/export_cells.py`` writes
them, and ``u``, one value per cell from left to right. Run this module to write
them again::

    python -m samurai_figures.pages.reaction_diffusion nagumo_ite_0.h5 nagumo_ite_100.h5

from ``docs/source/_ext``, in an environment with ``h5py``.
"""

import argparse
import importlib.util
import json
from functools import lru_cache
from math import ceil, exp, floor, log, sqrt
from pathlib import Path

from ..cells import check_disjoint, check_graded, load_cells
from ..draw import (
    HAIR,
    HEAVY,
    INK,
    INK2,
    PAPER,
    RED,
    RULE,
    bracket,
    callout,
    fig_label,
    line,
    patterns,
    rect,
    text,
)
from ..plots import Frame, axes, curve, level_rows, line_key, steps
from ..registry import PLATE_WIDTH, Drawing, figure

DATA = Path(__file__).resolve().parents[1] / "data"
SOURCE = "nagumo.cpp (demos/FiniteVolume), default options"

# The problem of the demo, as nagumo.cpp sets it with its default options.
A, B = -10.0, 10.0
D, K = 1.0, 10.0
Z0 = A / 5  # initial position of the wave
C = sqrt(K * D / 2)  # speed of the wave

# What the page says about each state; ``_state`` checks the data against it:
# the time, the file of the run, the number of cells of each level and the
# interval [a, b) the cells of level 8 cover.
STATES = {
    0: (0.0, "nagumo_ite_0.h5", {4: 4, 5: 6, 6: 8, 7: 18, 8: 76}, (-5.0, 0.9375)),
    1: (1.0, "nagumo_ite_100.h5", {4: 4, 5: 6, 6: 8, 7: 18, 8: 76}, (-2.8125, 3.125)),
}
# the front, as the legend gives it: where the exact solution goes from 0.99 to 0.01,
# rounded to 2 decimals, at t = 0 and at t = 1
EDGE = 0.01
FRONT = {0: (-4.06, 0.06), 1: (-1.82, 2.29)}
# the inset of each figure: this many cells of level 8 around the front, enlarged
# this many times in x
INSET_CELLS, ZOOM = 16, 5
# the window of each inset, as the legend gives it
WINDOW = {0.0: (-2.578125, -1.328125), 1.0: (-0.3125, 0.9375)}
# the values of u the inset marks on its right side
INSET_U = (0.25, 0.5, 0.75)

MINUS = "−"


def exact(x, t):
    """Return the exact traveling wave of nagumo.cpp at (x, t).

    ``exact_solution(x, t) = beta(x - c t)`` with ``beta(z) = e / (1 + e)`` and
    ``e = exp(-sqrt(k / (2 D)) (z - z0))``, written as ``1 / (1 + 1 / e)`` so that
    it does not overflow far left of the front.
    """
    s = sqrt(K / (2 * D)) * (x - C * t - Z0)
    return 1 / (1 + exp(s)) if s < 700 else 0.0


def front(t, value=0.5):
    """Return the abscissa where the exact solution equals ``value`` at time ``t``."""
    return Z0 + C * t + log(1 / value - 1) / sqrt(K / (2 * D))


class State:
    """One saved state of the run: its cells from left to right, and ``u`` on them.

    ``cells`` holds ``(level, k)``, the cell k of the level; ``u[n]`` is the value
    of the cell ``cells[n]``.
    """

    def __init__(self, exported, u, t):
        """Sort the cells of ``exported`` from left to right and pair them with ``u``."""
        self.h = exported.scaling_factor
        self.a = exported.origin[0]
        self.cells = sorted(((lv, k) for lv, k in exported.leaves), key=self.left)
        self.u = u
        self.t = t

    def left(self, cell):
        """Return the left end of ``cell``, in the coordinates of the program."""
        level, k = cell
        return self.a + k * self.h / 2**level

    def right(self, cell):
        """Return the right end of ``cell``."""
        level, k = cell
        return self.a + (k + 1) * self.h / 2**level


def _file(n):
    """Return the data file of the state ``n``."""
    return DATA / f"reaction_diffusion_t{n}.json"


@lru_cache(maxsize=2)
def _state(n):
    """Load the state ``n`` and check it: graded, covering [a, b), as the page says.

    The checks also place the front: the cells of level 8 must form one run that
    contains every x where the exact solution lies between 0.01 and 0.99.
    """
    t, _, counts_, finest_ = STATES[n]
    with open(_file(n), encoding="utf-8") as stream:
        u = json.load(stream)["u"]
    state = State(load_cells(_file(n)), u, t)
    # the cells as 2D cells of one row, for the checks of cells.py
    plane = [(level, k, 0) for level, k in state.cells]
    check_disjoint(plane)
    check_graded(plane)
    for left, right in zip(state.cells, state.cells[1:]):
        if abs(state.right(left) - state.left(right)) > 1e-12:
            raise ValueError(f"the cells {left} and {right} leave a gap or overlap")
    if state.left(state.cells[0]) != A or state.right(state.cells[-1]) != B:
        raise ValueError("the cells do not cover [-10, 10)")
    counts = {}
    for level, _ in state.cells:
        counts[level] = counts.get(level, 0) + 1
    if counts != counts_ or len(u) != len(state.cells):
        raise ValueError(f"the state at t = {t} has the cells {counts}")
    top = max(counts_)
    finest = [c for c in state.cells if c[0] == top]
    ks = [k for _, k in finest]
    if ks != list(range(ks[0], ks[0] + len(ks))):
        raise ValueError(f"the cells of level {top} at t = {t} are not one run")
    band = (state.left(finest[0]), state.right(finest[-1]))
    if band != finest_:
        raise ValueError(f"the cells of level {top} at t = {t} cover {band}")
    ends = (front(t, 1 - EDGE), front(t, EDGE))
    if tuple(round(e, 2) for e in ends) != FRONT[n]:
        raise ValueError(f"the front at t = {t} spans {ends}, not what the legend says")
    if not band[0] < ends[0] < ends[1] < band[1]:
        raise ValueError(f"the front at t = {t} leaves the cells of level {top}")
    return state


# --- the plate --------------------------------------------------------------------------


def _ticks():
    """Return the x ticks of the plate, every 5 units over [-10, 10]."""
    return [(v, str(v).replace("-", MINUS)) for v in range(-10, 11, 5)]


def _inset(state, fr, letter):
    """Draw the front of ``state`` enlarged ``ZOOM`` times in x, in the top right of ``fr``.

    The window holds the ``INSET_CELLS`` cells of level 8 around the middle of the
    front, where u and the exact wave, which agree to a line width in ``fr``, come
    apart: there the steps of u stand out on the wave. A thin rectangle marks the
    window on the curve of ``fr``, joined to the inset by a hairline; the callout
    ``letter`` marks the middle of the front inside the inset.
    """
    t = state.t
    h8 = (B - A) / 2**8
    first = int((front(t) - A) // h8) - INSET_CELLS // 2 + 1
    lo, hi = A + first * h8, A + (first + INSET_CELLS) * h8
    if (lo, hi) != WINDOW[t]:
        raise ValueError(f"the inset at t = {t} shows [{lo}, {hi})")
    cells = [(state.left(c), state.right(c), v) for c, v in zip(state.cells, state.u)]
    cells = [c for c in cells if lo - 1e-12 <= c[0] and c[1] <= hi + 1e-12]
    if len(cells) != INSET_CELLS or any(b - a != h8 for a, b, _ in cells):
        raise ValueError(f"the inset at t = {t} does not hold {INSET_CELLS} cells of level 8")
    w = ZOOM * (hi - lo) * fr.w / (B - A)
    ylim = (exact(hi, t) - 0.02, exact(lo, t) + 0.02)
    if not all(ylim[0] < v < ylim[1] for _, _, v in cells):
        raise ValueError(f"u leaves the inset at t = {t}")
    box = Frame(fr.X(9.85) - w, fr.Y(1.0), w, fr.Y(0.2) - fr.Y(1.0), xlim=(lo, hi), ylim=ylim)
    # the window on the main curve, and a hairline from its top right corner to the
    # top left corner of the inset, above the curve, which is under 0.25 there
    g = rect(
        fr.X(lo),
        fr.Y(ylim[1]),
        fr.X(hi) - fr.X(lo),
        fr.Y(ylim[0]) - fr.Y(ylim[1]),
        stroke=INK2,
        sw=HAIR,
    )
    g += line(fr.X(hi), fr.Y(ylim[1]), box.x, box.y, stroke=INK2, sw=HAIR)
    g += rect(box.x, box.y, box.w, box.h, fill=PAPER, stroke=INK2, sw=HAIR)
    g += curve(box, lambda x: exact(x, t))
    g += steps(box, cells, stroke=RED, sw=RULE)
    for q in range(floor(lo * 2), ceil(hi * 2) + 1):
        v = q / 2
        if lo < v < hi:
            g += line(box.X(v), box.bottom, box.X(v), box.bottom + 4, stroke=INK2, sw=HAIR)
            label = f"{v:g}".replace("-", MINUS)
            g += text(box.X(v), box.bottom + 15, label, anchor="middle", size=10.5)
    # the u axis of the inset, on its right side
    for v in INSET_U:
        if not ylim[0] < v < ylim[1]:
            raise ValueError(f"the tick u = {v} leaves the inset at t = {t}")
        g += line(box.right, box.Y(v), box.right + 4, box.Y(v), stroke=INK2, sw=HAIR)
        g += text(box.right + 7, box.Y(v) + 4, f"{v:g}", size=10.5)
    xf = front(t)
    g += callout(box.X(xf), box.Y(0.5), box.X(xf + 0.3), box.Y(0.66), letter, red=True)
    return g


def _panel(p, n, y, x0, w):
    """Draw the state ``n``: u over the exact wave, then the cells of each level.

    Return the markup and the y where the figure ends.
    """
    state = _state(n)
    t, top_level = state.t, max(STATES[n][2])
    fr = Frame(x0, y + 20, w, 150, xlim=(A, B), ylim=(-0.05, 1.05))
    yticks = [(0, "0"), (0.5, "0.5"), (1, "1")]
    g = axes(fr, xticks=_ticks(), yticks=yticks, xname="x", yname="u")
    # the steps of u in red on top of the heavy exact wave; at this scale they agree
    # to a line width, and the inset shows them apart
    cells = [(state.left(c), state.right(c), v) for c, v in zip(state.cells, state.u)]
    g += curve(fr, lambda x: exact(x, t))
    g += steps(fr, cells, stroke=RED, sw=RULE)
    g += line_key(
        fr.X(-9.3),
        fr.Y(0.55),
        [("exact solution", INK, HEAVY, ""), (f"u, {len(state.cells)} cells", RED, RULE, "")],
    )
    g += _inset(state, fr, "ab"[n])

    rows = fr.bottom + 42
    g += level_rows(x0, rows, w, state.cells, p)
    nrows = top_level - min(STATES[n][2]) + 1
    under = rows + (nrows - 1) * 30 + 16 + 12
    a, b = STATES[n][3]
    label = f"[{a:g}, {b:g})".replace("-", MINUS)
    g += bracket(fr.X(a), fr.X(b), under, label, color=RED, size=11)
    end = under + 44
    g += fig_label(x0 + w / 2, end, n + 1, f"t = {t:g}")
    return g, end + 14


@figure
def nagumo_result(p):
    """Draw u at t = 0 and t = 1 over the exact wave, and the levels of the cells.

    Plate of tutorial/reaction_diffusion.md. Fig. 1 is the state saved at t = 0,
    Fig. 2 the state at t = 1; each shows u, cell by cell, over the exact traveling
    wave, then the cells of each level on the same x axis.
    """
    x0, w = 96, 528
    first, split = _panel(p, 0, 0, x0, w)
    second, height = _panel(p, 1, split + 16, x0, w)
    s0, s1 = _state(0), _state(1)
    return Drawing(
        PLATE_WIDTH,
        height,
        "Two figures over x from -10 to 10, one above the other. Fig. 1, t = 0: u as"
        f" {len(s0.cells)} red steps over the exact traveling wave, a heavy curve that falls"
        f" from 1 to 0 around x = {front(0):g}; below, one row per level from level 4 to level"
        " 8, with 4, 6, 8, 18 and 76 cells; the 76 cells of level 8, hatched in red, cover"
        " [-5, 0.9375). In each plot a thin rectangle around the front is joined to an"
        " inset at the top right that enlarges it 5 times in x, over 16 cells of level 8,"
        " with u ticks 0.25, 0.5 and 0.75 on its right; there the red steps of u stand apart"
        " from the wave, and the letter a or b marks the middle of the front."
        " Fig. 2, t = 1: the same wave moved right, its middle at x ="
        f" {front(1):.3f}, u on {len(s1.cells)} cells; the 76 cells of level 8 cover"
        " [-2.8125, 3.125).",
        patterns(p),
        first + second,
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


def export(h5, out):
    """Write the cells and ``u`` of ``h5``, a file ``nagumo_ite_<n>.h5``, to ``out``.

    The cells go through ``export_cells.export``; ``u`` is read from the same
    file and sorted from left to right, rounded to 6 significant digits.
    """
    import h5py  # pylint: disable=import-outside-toplevel

    tools = _exporter()
    source = f"{SOURCE}, {Path(h5).name}"
    document = tools.export(h5, scaling_factor=B - A, origin=[A], source=source)
    with h5py.File(h5, "r") as f:
        points = f["mesh/points"][()]
        lefts = points[f["mesh/connectivity"][()]][:, :, 0].min(axis=1)
        values = f["mesh/fields/u"][()].tolist()
    u = [float(f"{v:.6g}") for _, v in sorted(zip(lefts.tolist(), values))]
    rows = ",\n".join(
        "    " + ", ".join(json.dumps(v) for v in u[q : q + 6]) for q in range(0, len(u), 6)
    )
    text_ = tools.dumps(document).rstrip().rstrip("}").rstrip()
    Path(out).write_text(f'{text_},\n  "u": [\n{rows}\n  ]\n}}\n', encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write data/reaction_diffusion_t<n>.json.")
    parser.add_argument("h5", type=Path, nargs=2, help="nagumo_ite_0.h5 and nagumo_ite_100.h5")
    for n_, h5_ in enumerate(parser.parse_args().h5):
        export(h5_, _file(n_))
