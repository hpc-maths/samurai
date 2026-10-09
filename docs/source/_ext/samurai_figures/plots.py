"""Plots of 1D fields: values over cells, as samurai computes them.

``draw.py`` draws cells; this module draws what a field holds in them. A ``Frame``
maps the coordinates of the data to drawing units, linear or logarithmic in y. On
it, ``steps`` draws cell averages, ``curve`` an analytic function or a polynomial
from ``lagrange``, ``detail`` the gap between a predicted and a real value,
``level_bars`` one value per level, and ``axes`` the ticks. ``level_rows`` draws the
cells of each level of a 1D mesh with their counts.

The numbers follow samurai: ``predict_children`` is the prediction of the library
with a stencil radius of 1 (order 3), ``parent_averages`` the field projection,
``level_threshold`` the threshold of the multiresolution at each level. Colors,
faces and strokes are those of ``draw.py``, so the plots follow the light and the
dark themes.
"""

from dataclasses import dataclass
from math import log10

from .draw import (
    FIG,
    HAIR,
    HEAVY,
    INK,
    INK2,
    RED,
    RULE,
    WASH,
    circle,
    line,
    num,
    rect,
    text,
)

MINUS = "−"


@dataclass(frozen=True)
class Frame:
    """The rectangle of a plot: ``(x, y)`` its top left corner, ``w`` by ``h`` units.

    ``xlim`` and ``ylim`` are the data ranges at its edges, y upwards. With
    ``ylog``, y is on a logarithmic scale and ``ylim`` must be positive.
    """

    x: float
    y: float
    w: float
    h: float
    xlim: tuple = (0.0, 1.0)
    ylim: tuple = (0.0, 1.0)
    ylog: bool = False

    @property
    def bottom(self):
        return self.y + self.h

    @property
    def right(self):
        return self.x + self.w

    def X(self, u):
        """The drawing abscissa of the data abscissa ``u``."""
        a, b = self.xlim
        return self.x + (u - a) / (b - a) * self.w

    def Y(self, v):
        """The drawing ordinate of the data value ``v``."""
        a, b = self.ylim
        if self.ylog:
            v, a, b = log10(v), log10(a), log10(b)
        return self.bottom - (v - a) / (b - a) * self.h


# --- numbers --------------------------------------------------------------------------


def predict_children(left, parent, right):
    """The values predicted for the two children of ``parent``, low child first.

    This is the prediction of samurai with a stencil radius of 1 (order 3): the
    children are ``parent -/+ (right - left) / 8``, ``left`` and ``right`` being the
    two neighbors of the parent on its level. The weights are
    ``prediction_coefficients<1>``, defined in
    include/samurai/numeric/prediction_coefficients.hpp, line 234, and solved
    there from the cell-average moment conditions: {1/8, 1, -1/8} for the low
    child, {-1/8, 1, 1/8} for the high one. ``compute_detail_op``
    (include/samurai/mr/operators.hpp, lines 158 to 172) subtracts this
    prediction from each child to get its detail.
    """
    slope = (right - left) / 8
    return parent - slope, parent + slope


def cell_averages(f, n, lo=0.0, hi=1.0, samples=40):
    """The averages of ``f`` over ``n`` equal cells of [lo, hi), by the midpoint rule."""
    h = (hi - lo) / n
    return [
        sum(f(lo + (k + (q + 0.5) / samples) * h) for q in range(samples)) / samples
        for k in range(n)
    ]


def parent_averages(averages):
    """The field projection to the next coarser level.

    Each parent is the mean of its two children.
    """
    return [(averages[2 * k] + averages[2 * k + 1]) / 2 for k in range(len(averages) // 2)]


def level_threshold(eps, level, max_level, dim=1):
    """The threshold on the details at ``level``: ``eps / 2^(dim (max_level - level))``.

    It is ``eps_l`` in include/samurai/mr/adapt.hpp, lines 324 and 325.
    """
    return eps / 2 ** (dim * (max_level - level))


def lagrange(points):
    """The polynomial through ``points``, a list of (x, y) with distinct x, as a function."""

    def p(u):
        total = 0.0
        for i, (xi, yi) in enumerate(points):
            term = yi
            for j, (xj, _) in enumerate(points):
                if j != i:
                    term *= (u - xj) / (xi - xj)
            total += term
        return total

    return p


def sci(v, digits=1, size=11):
    """``v`` in scientific notation for a text of font size ``size``: ``3.9·10^-6``.

    The exponent is raised and set smaller; the minus signs are true minus signs. A
    power of ten is written ``10^k`` alone.
    """
    mantissa, exponent = f"{v:.{digits}e}".split("e")
    power = (
        f'10<tspan baseline-shift="super" font-size="{num(0.72 * size)}">'
        f"{str(int(exponent)).replace('-', MINUS)}</tspan>"
    )
    if float(mantissa) == 1:
        return power
    return f"{mantissa.replace('-', MINUS)}\u00b7{power}"


# --- drawing --------------------------------------------------------------------------


def axes(fr, xticks=(), yticks=(), xname="", yname="", size=11):
    """The x axis along the bottom of ``fr`` and, with ``yticks``, the y axis on its left.

    A tick is ``(value, label)``; ``label`` is SVG markup, ``sci`` writes powers of
    ten. ``xname`` and ``yname`` name the axes in italics.
    """
    g = line(fr.x, fr.bottom, fr.right, fr.bottom)
    for u, label in xticks:
        g += line(fr.X(u), fr.bottom, fr.X(u), fr.bottom + 5)
        g += text(fr.X(u), fr.bottom + 18, label, anchor="middle", size=size)
    if xname:
        g += text(fr.right + 10, fr.bottom + 5, xname, size=14, fill=INK, cls="sm-fig-math")
    if yticks or yname:
        g += line(fr.x, fr.bottom, fr.x, fr.y - 6)
    for v, label in yticks:
        g += line(fr.x - 5, fr.Y(v), fr.x, fr.Y(v))
        g += text(fr.x - 8, fr.Y(v) + 4, label, anchor="end", size=size)
    if yname:
        g += text(fr.x, fr.y - 14, yname, anchor="middle", size=14, fill=INK, cls="sm-fig-math")
    return g


def log_ticks(lo, hi, size=11):
    """The ticks of the decades 10^lo to 10^hi, for ``axes`` on a frame with ``ylog``.

    ``size`` is the font size of the labels, the ``size`` of ``axes``.
    """
    return [(10.0**k, sci(10.0**k, size=size)) for k in range(lo, hi + 1)]


def steps(fr, cells, stroke=INK, sw=RULE, fills=None, joined=True):
    """Cell averages as steps: one flat segment per cell at its value.

    ``cells`` is a list of (a, b, v), the cell [a, b) holding the value v, in data
    coordinates and in order. ``joined`` links the steps with risers, as the graph
    of a piecewise constant function; without it each step stands alone. ``fills``,
    one per cell, fill the area under each step down to the bottom of the frame:
    a pattern of ``draw.patterns`` (``url(#p-k)``, ``url(#p-r)``), or ``None``.
    """
    g = ""
    if fills:
        for (a, b, v), fill in zip(cells, fills):
            if fill:
                top = fr.Y(v)
                g += (
                    f'<rect x="{num(fr.X(a))}" y="{num(top)}" width="{num(fr.X(b) - fr.X(a))}"'
                    f' height="{num(fr.bottom - top)}" fill="{fill}"/>'
                )
    if joined:
        d = "".join(
            f"{'L' if q else 'M'}{num(fr.X(a))} {num(fr.Y(v))}H{num(fr.X(b))}"
            for q, (a, b, v) in enumerate(cells)
        )
        g += f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}"/>'
    else:
        for a, b, v in cells:
            g += line(fr.X(a), fr.Y(v), fr.X(b), fr.Y(v), stroke=stroke, sw=sw)
    return g


def curve(fr, f, lo=None, hi=None, stroke=INK, sw=HEAVY, dash="", samples=400):
    """The graph of the function ``f`` from ``lo`` to ``hi``, by default across ``fr``.

    An analytic solution, or a polynomial from ``lagrange``. It is drawn heavy, the
    weight of the outline a figure compares against. ``f`` is sampled at
    ``samples + 1`` points, and the graph stops where it leaves ``ylim``, so that
    it stays inside the frame.
    """
    lo = fr.xlim[0] if lo is None else lo
    hi = fr.xlim[1] if hi is None else hi
    bottom, top = fr.ylim
    d, pen = "", False
    for q in range(samples + 1):
        u = lo + (hi - lo) * q / samples
        v = f(u)
        inside = bottom <= v <= top
        if inside:
            d += f"{'L' if pen else 'M'}{num(fr.X(u))} {num(fr.Y(v))}"
        pen = inside
    dashes = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}"'
        f' stroke-linejoin="round"{dashes}/>'
    )


def dots(fr, points, color=INK, r=2.6):
    """Filled dots at ``points``, a list of (x, y): the data a polynomial goes through."""
    return "".join(circle(fr.X(u), fr.Y(v), r, fill=color) for u, v in points)


def detail(fr, a, b, predicted, real, label, p, inset=4):
    """The detail of the cell [a, b): its predicted value, its real value, the gap.

    The predicted value is a dashed bar in secondary ink, the real value a heavy red
    bar, and a red double arrow at the middle of the cell spans the gap, named
    ``label`` (SVG markup, ``d<tspan ...>1</tspan>``). The bars stop ``inset``
    units short of the cell edges, so that neighboring cells stay apart. The arrow
    heads are the ``{p}-ar`` marker of ``draw.patterns``; the gap must be at least
    14 units for both heads to show.
    """
    xa, xb = fr.X(a) + inset, fr.X(b) - inset
    xm, yp, yr = (xa + xb) / 2, fr.Y(predicted), fr.Y(real)
    heads = f'marker-start="url(#{p}-ar)" marker-end="url(#{p}-ar)"'
    return (
        line(xa, yp, xb, yp, stroke=INK2, dash="4 3")
        + line(xa, yr, xb, yr, stroke=RED, sw=HEAVY)
        + line(xm, yp, xm, yr, stroke=RED, extra=heads)
        + text(xm + 7, (yp + yr) / 2 + 4, label, size=15, fill=RED, cls="sm-fig-math")
    )


def line_key(x, y, entries, pitch=22, size=12):
    """A key of line styles, one entry a row: a short sample line and its meaning.

    An entry is ``(label, stroke, sw, dash)``; a red line gets a red label.
    """
    g = ""
    for r, (label, stroke, sw, dash) in enumerate(entries):
        yy = y + r * pitch
        g += line(x, yy, x + 22, yy, stroke=stroke, sw=sw, dash=dash)
        g += text(x + 30, yy + 4, label, size=size, fill=RED if stroke == RED else INK2)
    return g


def level_rows(x, y, w, leaves, p, pitch=30, h=16, size=12):
    """The cells of each level of a 1D mesh of [0, 1), one row a level, and their count.

    ``leaves`` is a list of (level, k), the cell k of ``level`` covering
    [k / 2^level, (k + 1) / 2^level). The rows go from the coarsest level at the top
    to the finest, whose cells are hatched in red; a faint dashed line runs where a
    level has no cell. The row of a level is ``h`` units tall, its top at
    ``y + (level - min level) * pitch``.
    """
    levels = sorted({level for level, _ in leaves})
    g = ""
    for r, level in enumerate(range(levels[0], levels[-1] + 1)):
        yy, cw, red = y + r * pitch, w / 2**level, level == levels[-1]
        ks = [k for lv, k in leaves if lv == level]
        g += line(x, yy + h, x + w, yy + h, stroke=FIG, dash="3 3")
        for k in ks:
            if red:
                g += rect(x + k * cw, yy, cw, h, fill=f"url(#{p}-r)", stroke=RED, sw=HAIR)
            else:
                g += rect(x + k * cw, yy, cw, h, fill=WASH, stroke=INK)
        tone = RED if red else INK
        g += text(x - 12, yy + 12, f"level {level}", anchor="end", size=size, fill=tone)
        count = f"{len(ks)} cell{'' if len(ks) == 1 else 's'}"
        g += text(x + w + 12, yy + 12, count, size=size - 0.5, fill=RED if red else INK2)
    return g


def level_bars(fr, values, p, width=34, size=11.5):
    """One bar per level, its height a value: a threshold, an error, a count.

    ``values`` maps each level to its value; the bars stand on the bottom of
    ``fr``, spread evenly across it, with the level under each bar and the value,
    by ``sci``, over it. The finest level is hatched in red. On a frame with
    ``ylog`` the bottom of the frame is ``ylim[0]``, not zero.
    """
    levels = sorted(values)
    pitch = fr.w / len(levels)
    g = ""
    for q, level in enumerate(levels):
        cx, top, red = fr.x + (q + 0.5) * pitch, fr.Y(values[level]), level == levels[-1]
        fill, stroke, sw = (f"url(#{p}-r)", RED, HAIR) if red else (WASH, INK, RULE)
        g += rect(cx - width / 2, top, width, fr.bottom - top, fill=fill, stroke=stroke, sw=sw)
        tone = RED if red else INK
        g += text(cx, fr.bottom + 17, f"level {level}", anchor="middle", size=size, fill=tone)
        small = size - 0.5
        label = sci(values[level], size=small)
        g += text(cx, top - 7, label, anchor="middle", size=small, fill=RED if red else INK2)
    return g
