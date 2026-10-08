"""Processes drawn as steps: a cycle of stations, a chain of boxes.

A station is one step of a process drawn as a small figure (its art), with its
name and the statement that performs it in the mono under the name, as in the
time step plate of the getting started tutorial. ``cycle`` sets stations in a row,
links each one to the next with an arrow and, given a ``Loop``, returns from the
last to the first under a note. ``chain`` sets boxes of text in a row, with a label
on each arrow between them.

Both return a ``Flow``: the SVG of the labels and the arrows, the height it needs,
and one panel per step for the narrow screens. A panel holds one step and none of
the arrows that link it to its neighbors (a chain panel holds its own outgoing
arrow whole), so a plate stacked on a phone shows no stray piece of a connector,
and all the steps of a flow come out at the same scale. A panel is only clean if
the art, the name and the code of each step fit the width of its box.

The statements are set in the mono glyph by glyph, each at its own x, 0.6 em
apart. Set as a plain run, Chrome rounds the advance of each glyph to whole pixels
at the size the drawing is shown at, so that the run comes out between 0.56 and
0.64 em a glyph depending on the screen; placed glyph by glyph it is 0.6 em a glyph
at any size, and ``text_width`` gives its exact width.

The arrows use the arrow head ``{p}-a`` of ``draw.patterns(p)``, which the defs of
the drawing must hold.
"""

import re
from dataclasses import dataclass
from html import escape, unescape

from .arrays import MONO_ADVANCE
from .draw import INK, INK2, PAPER, fig_label, line, num, path, rect, text

# Under the art of a station, the baseline of its name; under the name, the
# baseline of its code. The return of a cycle leaves just under the code and runs
# RETURN_DY under it.
LABEL_DY, CODE_DY, RETURN_DY = 40, 20, 44
# The room a station panel keeps above the art, for a callout, and under the
# baseline of the code, for its descenders.
HEAD, FOOT = 22, 8
# The panel of the note of a loop stops this far inside the paper band of the note,
# so that the return line, which meets the band at its two ends, leaves no speck at
# the edges of the panel once it is scaled down on a phone.
NOTE_INSET = 2

# The face of the text in a box: CSS class, size and color.
FACES = {
    "label": ("sm-fig-label", 12, INK),
    "math": ("sm-fig-math", 15, INK),
    "code": ("sm-fig-code", 11, INK),
}


@dataclass(frozen=True)
class Station:
    """A step of a cycle: ``name`` labels it, ``code`` is its statement, in the mono."""

    name: str
    code: str = ""


@dataclass(frozen=True)
class Row:
    """Where the stations of a cycle stand, in drawing units.

    Station ``k`` is centered on ``x0 + k * pitch``. Its box is ``box`` wide; its
    art is drawn by the figure from ``top`` down to ``top + art``, inside the box.
    The arrow to the next station runs at mid-height of the art, from ``gap``
    after the edge of the box to ``gap`` before the next one. ``first`` is the
    number of the first station in "Fig. N."; with ``None`` the stations carry
    their name alone.
    """

    x0: float
    pitch: float
    box: float
    top: float
    art: float = 92
    gap: float = 6
    first: int | None = 1

    def center(self, k):
        """Return the abscissa of the center of station ``k``."""
        return self.x0 + k * self.pitch

    @property
    def label_y(self):
        """The baseline of the station names."""
        return self.top + self.art + LABEL_DY

    @property
    def code_y(self):
        """The baseline of the statements under the names."""
        return self.label_y + CODE_DY

    def link(self, k):
        """Return the ends ``(x1, x2, y)`` of the arrow from station ``k`` to the next."""
        cx = self.center(k)
        x1, x2 = cx + self.box / 2 + self.gap, cx + self.pitch - self.box / 2 - self.gap
        return x1, x2, self.top + self.art / 2

    def region(self, k):
        """Return the panel of station ``k``: its box, from above the art to under the code."""
        y = self.top - HEAD
        return (self.center(k) - self.box / 2, y, self.box, self.code_y + FOOT - y, "half")


@dataclass(frozen=True)
class Loop:
    """The return of a cycle, from its last station to its first.

    ``note`` is SVG markup, set in the serif italic on a paper band ``width``
    wide that interrupts the return line; by default the band is the estimated
    width of the note, plus 12 units on each side.
    """

    note: str
    width: float | None = None


@dataclass(frozen=True)
class Lane:
    """Where the boxes of a chain stand, in drawing units.

    The first box starts at ``x`` and the last one ends at ``x + width``; each box
    is ``box`` wide and ``height`` tall, its top at ``y``. Each arrow keeps ``gap``
    from the boxes it links.
    """

    x: float
    y: float
    width: float
    box: float
    height: float = 30
    gap: float = 6

    @property
    def axis(self):
        """The ordinate of the arrows, at mid-height of the boxes."""
        return self.y + self.height / 2

    def pitch(self, n):
        """Return the distance between the left edges of two boxes in a chain of ``n``."""
        return (self.width - self.box) / (n - 1) if n > 1 else self.box + 2 * self.gap


@dataclass(frozen=True)
class Step:
    """A box of a chain, with ``code``, its statement, in the mono under it.

    ``text`` is SVG markup, which may hold ``<tspan>`` elements, set in ``face``:
    ``"label"`` (the sans), ``"math"`` (the serif italic) or ``"code"`` (the mono);
    with ``"code"``, ``text`` is plain text, set glyph by glyph.
    """

    text: str
    face: str = "label"
    code: str = ""


@dataclass(frozen=True)
class Flow:
    """The labels and the arrows of a flow, ``height`` units tall from the top of the drawing.

    ``panels`` cut it into its steps for the narrow screens, in the form of
    ``Drawing.panels``.
    """

    body: str
    height: float
    panels: tuple


def cycle(p, row, stations, loop=None):
    """Draw the names, codes and arrows of ``stations`` standing on ``row``.

    The figure draws the art of each station itself, at ``row.center(k)``. With a
    ``loop``, an arrow returns from under the last station to under the first,
    through the note of the loop; the note gets a panel of its own. Raise
    ``ValueError`` if the code of a station is wider than the box of the row.
    """
    for station in stations:
        if text_width(station.code) > row.box:
            raise ValueError(
                f"{station.code!r} is {num(text_width(station.code))} units wide in the mono,"
                f" the box of the row only {num(row.box)}"
            )
    arrow = f'marker-end="url(#{p}-a)"'
    g = ""
    for k, station in enumerate(stations):
        g += _station(row, k, station)
        if k < len(stations) - 1:
            x1, x2, y = row.link(k)
            g += line(x1, y, x2, y, stroke=INK2, extra=arrow)
    panels = tuple(row.region(k) for k in range(len(stations)))
    if loop is None:
        return Flow(g, row.code_y + FOOT, panels)
    return_, height, panel = _return(arrow, row, len(stations), loop)
    return Flow(g + return_, height, (*panels, panel))


def _return(arrow, row, n, loop):
    """Draw the return of a cycle of ``n`` stations; give its height and the panel of its note."""
    first, last, yb = row.center(0), row.center(n - 1), row.code_y + RETURN_DY
    g = path(
        f"M{num(last)} {num(row.code_y + 12)} V{num(yb)} H{num(first)} V{num(row.code_y + 16)}",
        extra=arrow,
    )
    # the band hides the return line under the note, and the panel of the note
    # stays inside the band, so that no piece of the line shows beside it
    mid = (first + last) / 2
    w = loop.width or text_width(loop.note, "math", 14) + 24
    g += rect(mid - w / 2, yb - 12, w, 22, fill=PAPER, stroke="none")
    g += text(mid, yb + 5, loop.note, anchor="middle", size=14, fill=INK, cls="sm-fig-math")
    return g, yb + 16, (mid - w / 2 + NOTE_INSET, yb - 22, w - 2 * NOTE_INSET, 36, "")


def _station(row, k, station):
    """Draw the name of station ``k``, with its number if the row has one, and its code."""
    cx = row.center(k)
    if row.first is None:
        g = text(cx, row.label_y, station.name, "middle", 14, INK, "sm-fig-math")
    else:
        g = fig_label(cx, row.label_y, row.first + k, station.name)
    if station.code:
        g += _mono(cx, row.code_y, station.code)
    return g


def chain(p, lane, steps, links):
    """Draw the boxes of ``steps`` on ``lane``, ``links[k]`` over the arrow after box ``k``.

    ``links`` holds one label per arrow, SVG markup set in the serif italic, or
    ``""`` for an arrow without label. Each panel holds a box and the whole arrow
    that leaves it, so that the stacked panels still read as a chain; all the
    panels have the same width.
    """
    if len(links) != len(steps) - 1:
        raise ValueError(f"{len(steps)} steps need {len(steps) - 1} links, not {len(links)}")
    arrow = f'marker-end="url(#{p}-a)"'
    pitch = lane.pitch(len(steps))
    # the labels of the arrows above, the codes of the boxes below
    top, bottom = lane.y - 26, lane.y + lane.height + (24 if any(s.code for s in steps) else 6)
    g, panels = "", []
    for k, step in enumerate(steps):
        x = lane.x + k * pitch
        g += _box(lane, x, step)
        if k < len(steps) - 1:
            x1, x2 = x + lane.box + lane.gap, x + pitch - lane.gap
            g += line(x1, lane.axis, x2, lane.axis, stroke=INK2, extra=arrow)
            if links[k]:
                g += text((x1 + x2) / 2, lane.axis - 7, links[k], "middle", 14, INK, "sm-fig-math")
        # from half a gap before the box to half a gap before the next box: the
        # arrow that comes in stays out, the arrow that leaves is in whole
        panels.append((x - lane.gap / 2, top, pitch, bottom - top, "half"))
    return Flow(g, bottom, tuple(panels))


def _box(lane, x, step):
    """Draw the box of ``step`` on ``lane``, its left edge at ``x``, and its code under it."""
    cls, size, fill = FACES[step.face]
    cx = x + lane.box / 2
    g = rect(x, lane.y, lane.box, lane.height, fill=PAPER, stroke=INK)
    if step.face == "code":
        g += _mono(cx, lane.axis + 0.35 * size, step.text, size, fill)
    else:
        g += text(cx, lane.axis + 0.35 * size, step.text, "middle", size, fill, cls)
    if step.code:
        g += _mono(cx, lane.y + lane.height + 18, step.code)
    return g


def _mono(cx, y, s, size=10.5, fill=INK2):
    """Set the plain text ``s`` in the mono, centered on ``cx``, each glyph at its own x.

    Each glyph is centered in its slot of 0.6 em: whatever the advance the browser
    rounds it to, the run stays centered on ``cx``.
    """
    advance = MONO_ADVANCE * size
    x0 = cx - advance * len(s) / 2
    xs = " ".join(num(x0 + (k + 0.5) * advance) for k in range(len(s)))
    spans = f'<tspan x="{xs}">{escape(s, quote=False)}</tspan>'
    return text(cx, y, spans, anchor="middle", size=size, fill=fill, cls="sm-fig-code")


def text_width(markup, face="code", size=10.5):
    """Return the width of ``markup`` set in ``face`` at ``size``, tags left out.

    Exact for the mono as flows set it, glyph by glyph 0.6 em apart; for the sans
    and the serif italic, an estimate at 0.56 em per glyph. A figure uses it to
    choose a box that holds the names and the codes of its steps.
    """
    em = MONO_ADVANCE if face == "code" else 0.56
    return em * size * len(unescape(re.sub(r"<[^>]*>", "", markup)))
