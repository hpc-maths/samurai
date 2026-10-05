"""The registry of the figures, and the drawing a figure function returns."""

from dataclasses import dataclass

from .draw import svg

FIGURES = {}


def figure(function):
    """Register a figure function under its name, for the ``:figure:`` option."""
    FIGURES[function.__name__] = function
    return function


@dataclass(frozen=True)
class Drawing:
    """An SVG drawing of ``width`` by ``height`` units.

    ``label`` describes the drawing to screen readers. ``defs`` and ``body`` are
    SVG markup. ``panels`` cut a plate into its figures on narrow screens: each
    panel is ``(x, y, w, h, kind)`` in drawing units, ``kind`` being ``"half"``
    for a figure that fits half the width of the screen, ``"wide"`` for one that
    stays wider than the screen and scrolls, ``""`` otherwise.
    """

    width: float
    height: float
    label: str
    defs: str
    body: str
    panels: tuple = ()

    def svg(self, id_=""):
        return svg(self.width, self.height, self.label, self.defs, self.body, id_)
