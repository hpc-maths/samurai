"""The registry of the figures, and the drawing a figure function returns.

A figure function takes the prefix of the ids of its SVG elements and returns a
``Drawing``. Diagrams are drawn ``DIAGRAM_WIDTH`` units wide, the text column;
plates ``PLATE_WIDTH`` units wide, the inside of a plate frame. The ``plate`` and
``diagram`` directives (``_ext/plates.py``) refer to a figure by the name of its
function.
"""

import importlib
import pkgutil
from dataclasses import dataclass

from .draw import svg

PLATE_WIDTH = 711
DIAGRAM_WIDTH = 680

FIGURES = {}


def figure(function):
    """Register a figure function under its name, for the ``:figure:`` option."""
    FIGURES[function.__name__] = function
    return function


def load_pages():
    """Import every module of ``pages/``, which registers the figures it defines.

    A page module needs no entry anywhere else: adding the file is enough.
    """
    pages = importlib.import_module(f"{__package__}.pages")
    for module in pkgutil.iter_modules(pages.__path__):
        importlib.import_module(f"{pages.__name__}.{module.name}")


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
