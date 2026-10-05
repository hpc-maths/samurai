"""Figures of the samurai documentation, drawn from code.

Each figure is a function registered with ``@figure`` in the module of its page,
``pages/<page>.py``; the ``plate`` and ``diagram`` directives of
``_ext/plates.py`` call it by name and put its SVG in the page. The drawing
primitives that pages share are modules of the package, such as ``draw.py`` and
``mesh.py``. A figure is true to samurai: meshes are graded quadtrees built and
checked by ``mesh.py``, intervals are half-open, and the cells drawn for a program
are the cells the program prints.
"""

from .registry import FIGURES, Drawing, figure, load_pages

load_pages()

__all__ = ["FIGURES", "Drawing", "draw", "figure"]


def draw(name, prefix):
    """The drawing of the figure ``name``, its SVG ids starting with ``prefix``.

    Raise ``KeyError`` for an unknown figure.
    """
    return FIGURES[name](prefix)
