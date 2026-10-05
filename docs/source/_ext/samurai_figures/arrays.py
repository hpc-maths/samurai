"""How samurai stores things: arrays, printed intervals and intervals of step 2.

``Array`` places a row of boxed entries, ``draw_array`` draws it with the position
of each entry, and ``connector`` links a cell or an entry to another entry. A
printed interval, such as ``0->[14,16)@-6:1``, is a token row: ``interval_token``
splits it as samurai prints it and ``token_row`` sets it in the mono with a label
under each part. ``even_elements`` and ``odd_elements`` compute the intervals of
step 2 that samurai returns, and ``strided_row`` draws one over its cells. Colours,
faces and strokes are those of ``draw.py``.
"""

from dataclasses import dataclass

from .draw import (
    EMPTY,
    FIG,
    HAIR,
    INK,
    INK2,
    INK3,
    PAPER,
    RED,
    RULE,
    RWASH,
    WASH,
    bracket,
    circle,
    num,
    path,
    rect,
    text,
)

# The advance of a glyph of Source Code Pro, in em: 600 units of 1000.
MONO_ADVANCE = 0.6


def mono_width(s, size):
    """The width of the string ``s`` set in the mono at ``size``."""
    return MONO_ADVANCE * size * len(s)


@dataclass(frozen=True)
class Array:
    """The place of a row of ``n`` boxed entries, ``w`` by ``h``, the first at (x, y).

    The entries touch: entry k spans [x + k w, x + (k + 1) w). The methods give
    the points where a connector meets an entry.
    """

    x: float
    y: float
    w: float
    n: int
    h: float = 24

    def left(self, k):
        """The left edge of entry ``k``."""
        return self.x + k * self.w

    def cx(self, k):
        """The centre of entry ``k`` along x."""
        return self.x + (k + 0.5) * self.w

    @property
    def right(self):
        """The right edge of the last entry."""
        return self.x + self.n * self.w

    @property
    def bottom(self):
        """The lower edge of the entries."""
        return self.y + self.h


def draw_array(
    a, values=None, red=(), name="", name_cls="sm-fig-code", positions=True, groups=(), size=11
):
    """Draw the array ``a`` with ``values`` in its entries, in the mono.

    ``values`` holds one string or number per entry; ``None`` leaves the entries
    empty. The entries in ``red`` are the ones the figure is about: red wash, red
    outline, red text. ``name`` labels the array on its left, in the mono by default
    (``name_cls="sm-fig-math"`` for a variable such as *u*). ``positions`` writes
    the position of each entry under it, from 0. ``groups`` are spans
    ``(first, last, label)`` of entries, marked under the positions with a hairline
    and labelled in the sans, such as the entries of one level.

    Raise ``ValueError`` when a value does not fit in its entry.
    """
    if values is not None and len(values) != a.n:
        raise ValueError(f"the array has {a.n} entries but {len(values)} values")
    red = set(red)
    g = ""
    # the red entries last, so that their outline is not covered by a neighbour's
    for k in sorted(range(a.n), key=lambda k: k in red):
        tone = k in red
        g += rect(
            a.left(k), a.y, a.w, a.h, fill=RWASH if tone else PAPER, stroke=RED if tone else INK
        )
        if values is not None and values[k] != "":
            s = str(values[k])
            if mono_width(s, size) > a.w - 4:
                raise ValueError(f"the value {s!r} does not fit in an entry {a.w} units wide")
            g += text(
                a.cx(k),
                a.y + a.h / 2 + 0.36 * size,
                s,
                anchor="middle",
                size=size,
                fill=RED if tone else INK,
                cls="sm-fig-code",
            )
    y = a.bottom
    if positions:
        for k in range(a.n):
            g += text(
                a.cx(k), y + 14, k, anchor="middle", size=10.5, fill=RED if k in red else INK2
            )
        y += 14
    for first, last, label in groups:
        # a hairline under the entries first to last, with ticks at both ends
        xa, xb, yg = a.left(first) + 3, a.left(last + 1) - 3, y + 9
        g += path(
            f"M{num(xa)} {num(yg - 4)}V{num(yg)}H{num(xb)}V{num(yg - 4)}", stroke=INK3, sw=HAIR
        )
        g += text((xa + xb) / 2, yg + 14, label, anchor="middle", size=11, fill=INK2)
    if name:
        cls_size = 14 if name_cls == "sm-fig-math" else 12
        g += text(
            a.x - 12,
            a.y + a.h / 2 + 0.36 * cls_size,
            name,
            anchor="end",
            size=cls_size,
            fill=INK,
            cls=name_cls,
        )
    return g


def connector(x1, y1, x2, y2, red=False, arrow=""):
    """A line from (x1, y1) to (x2, y2) that leaves and arrives vertically.

    It links a cell to the entry that stores it, or an entry to the entry it
    points to. A dot marks the start. ``red`` draws it in red, for the link the
    figure is about. ``arrow`` is the id of an arrow head from ``draw.patterns``
    (``f"{p}-a"`` or ``f"{p}-ar"``), drawn at the end.
    """
    color = RED if red else INK3
    ym = (y1 + y2) / 2
    d = (
        f"M{num(x1)} {num(y1)}C{num(x1)} {num(ym)} {num(x2)} {num(ym)} {num(x2)} {num(y2)}"
        if x1 != x2
        else f"M{num(x1)} {num(y1)}V{num(y2)}"
    )
    extra = f'marker-end="url(#{arrow})"' if arrow else ""
    return path(d, stroke=color, sw=RULE, extra=extra) + circle(x1, y1, 1.8, fill=color)


# The labels of the parts of a printed interval, see interval_token.
TOKEN_PARTS = ("position", "start", "end", "index", "step")


def interval_token(start, end, index, step=1, position=None, red=()):
    """The parts of an interval as samurai prints it: ``[start,end)@index:step``.

    This is the format of ``operator<<`` in ``include/samurai/interval.hpp``; a
    ``position`` adds the ``position->`` that a level cell array prints before each
    interval, its place in the array of intervals. The result is a list of
    ``(text, label, color)`` for ``token_row``: the numbers carry the labels of
    ``TOKEN_PARTS``, the punctuation none. The parts named in ``red`` are red.
    """
    red = set(red)
    unknown = red - set(TOKEN_PARTS)
    if unknown:
        raise ValueError(f"unknown parts {sorted(unknown)}, expected some of {TOKEN_PARTS}")

    def part(value, label):
        return (str(value), label, RED if label in red else INK)

    parts = []
    if position is not None:
        parts += [part(position, "position"), ("->", None, INK3)]
    parts += [
        ("[", None, INK3),
        part(start, "start"),
        (",", None, INK3),
        part(end, "end"),
        (")@", None, INK3),
        part(index, "index"),
        (":", None, INK3),
        part(step, "step"),
    ]
    return parts


def token_width(parts, size=18):
    """The width of the token row of ``parts`` at ``size``."""
    return mono_width("".join(s for s, _, _ in parts), size)


def token_row(x, y, parts, size=18, anchor="start", label_size=11):
    """Set ``parts`` in the mono on the baseline y, with a label under each part.

    ``parts`` is a list of ``(text, label, color)``, such as ``interval_token``
    returns. A labelled part gets a hairline brace under its characters and its
    label in the sans, in its colour. A label that would touch the previous one
    drops to a second line, with a longer leader. ``anchor`` is ``"start"``,
    ``"middle"`` or ``"end"`` of the row at x.
    """
    advance = MONO_ADVANCE * size
    width = token_width(parts, size)
    x0 = {"start": x, "middle": x - width / 2, "end": x - width}[anchor]
    # Each glyph gets its own x: browsers round the advance of a glyph to whole
    # pixels, and the error would add up along the row and shift the braces.
    spans, k = "", 0
    for s, _, color in parts:
        xs = " ".join(num(x0 + (k + i) * advance) for i in range(len(s)))
        spans += f'<tspan x="{xs}" fill="{color}">{s}</tspan>'
        k += len(s)
    g = text(x0, y, spans, size=size, fill=INK, cls="sm-fig-code")
    brace = y + 0.3 * size
    rows_end = [float("-inf"), float("-inf")]
    k = 0
    for s, label, color in parts:
        xa, xb = x0 + k * advance, x0 + (k + len(s)) * advance
        k += len(s)
        if not label:
            continue
        xm, half = (xa + xb) / 2, 0.29 * label_size * len(label) + 3
        row = 0 if xm - half >= rows_end[0] else 1
        rows_end[row] = xm + half
        ly = brace + 18 + row * 15
        g += path(
            f"M{num(xa + 1)} {num(brace)}v3H{num(xb - 1)}v-3M{num(xm)} {num(brace + 3)}"
            f"V{num(ly - label_size + 2)}",
            stroke=color,
            sw=HAIR,
        )
        g += text(xm, ly, label, anchor="middle", size=label_size, fill=color)
    return g


def even_elements(start, end):
    """The even cells of [start, end), as samurai's ``Interval::even_elements``.

    Return ``(start, end, 2)``: the start moves to the first even cell, and the end
    to one past the last even cell. Return ``None`` when no cell is even, where
    samurai returns an interval that ``is_valid`` rejects.
    """
    s = start + (1 if start & 1 else 0)
    e = end - (0 if end & 1 else 1)
    return (s, e, 2) if s < e else None


def odd_elements(start, end):
    """The odd cells of [start, end), as samurai's ``Interval::odd_elements``.

    Return ``(start, end, 2)``, or ``None`` when no cell is odd.
    """
    s = start + (0 if start & 1 else 1)
    e = end - (1 if end & 1 else 0)
    return (s, e, 2) if s < e else None


def strided_row(x0, y, cw, to, interval, ch=26, start=0, tone="red", label=None, hops=True):
    """The cells ``start`` to ``to - 1`` with an interval of step 2 over them.

    ``interval`` is ``(a, b, step)``, as ``even_elements`` returns. The cells
    a, a + step, ... below b are filled, in red (``tone="red"``) or in ink
    (``tone="ink"``); the cells between them and outside are faint. ``hops`` draws
    an arc over the row from each cell of the interval to the next. Under the row,
    the half-open bracket of ``draw.bracket`` spans [a, b), labelled ``label``,
    ``"[a, b), step 2"`` by default.
    """
    a, b, step = interval
    members = set(range(a, b, step))
    fill, stroke = (RWASH, RED) if tone == "red" else (WASH, INK)
    g = ""
    for k in range(start, to):
        if k not in members:
            g += rect(x0 + (k - start) * cw, y, cw, ch, stroke=FIG, sw=EMPTY)
    for k in sorted(members):
        g += rect(x0 + (k - start) * cw, y, cw, ch, fill=fill, stroke=stroke)
    if hops:
        for k in sorted(members)[:-1]:
            xa, xb = x0 + (k - start + 0.5) * cw, x0 + (k + step - start + 0.5) * cw
            g += path(
                f"M{num(xa)} {num(y - 3)}Q{num((xa + xb) / 2)} {num(y - 3 - 0.55 * (xb - xa))}"
                f" {num(xb)} {num(y - 3)}",
                stroke=stroke,
                sw=HAIR,
            )
    if label is None:
        label = f"[{a}, {b}), step {step}"
    g += bracket(x0 + (a - start) * cw, x0 + (b - start) * cw, y + ch + 9, label, color=stroke)
    return g
