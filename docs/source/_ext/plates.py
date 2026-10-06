"""Plates and diagrams, drawn from code by the figures of ``samurai_figures``.

Two directives, written in MyST as fenced blocks::

    ```{plate} The disc transported by the velocity (1, 1)
    :figure: transported_disc
    :label: plate-transport

    **Fig. 1.** The initial state. *a*, the finest cells follow the edge of the disc.

    *The plate is drawn with fewer levels than the program.*
    ```

    ```{diagram}
    :figure: boolean_operations

    The four results on the cells 0 to 13 of level 0.
    ```

A plate is a full-width illustration in a double frame, with its title in spaced
capitals and "Plate 1" on the right, then its legend under "Explanation of Plate 1",
in two columns (``:columns: 1`` for one). In the legend, "**Fig. 1.**" at the start of
a paragraph names a figure of the plate, a single italic letter names a callout, and
a paragraph entirely in italics is a note. The plates of a page are numbered
"Plate 1", "Plate 2"; ``:label:`` makes a plate a target for ``{ref}`label```, which
prints "Plate 1 (title)". On narrow screens the figures of a plate are stacked, one
under the other, from the panels of its figure function.

A diagram is an inline drawing with its caption. It is not numbered; a ``:label:``
is linked with an explicit text, ``{ref}`the diagram <label>```.

``:figure:`` names a function registered in ``samurai_figures``. A schematic that
depends on no program data may instead be a hand-written SVG file, ``:svg:`` with its
path relative to the page: ``samurai_figures/static_svg.py`` checks it against the
rules of the figures and inlines it like a generated drawing. Its plate panels come
from ``:panels:`` or from ``data-panels`` on its root ``<svg>``, as
``x y w h [half|wide]`` separated by semicolons. Builders other than HTML get the
legend or the caption only.
"""

import re
from html import escape
from pathlib import Path

import samurai_figures
from docutils import nodes
from docutils.parsers.rst import directives
from samurai_figures import FIGURES, draw
from samurai_figures.registry import DIAGRAM_WIDTH, PLATE_WIDTH
from samurai_figures.static_svg import THEME_CSS, load
from sphinx.transforms import SphinxTransform
from sphinx.util.docutils import SphinxDirective

FIG_LABEL = re.compile(r"Figs?\. \d+")
# Every module and data file of the package, the page modules of pages/ included, so
# that a new file is tracked without being listed here: a cell list re-exported by a
# program rebuilds the pages that draw it.
FIGURE_SOURCES = sorted(
    str(source)
    for pattern in ("*.py", "*.json", "*.svg")
    for source in Path(samurai_figures.__file__).parent.rglob(pattern)
)


class plate(nodes.General, nodes.Element):
    """A plate: its drawing is in the attributes, its children are the legend."""


class diagram(nodes.General, nodes.Element):
    """A diagram: its drawing is in the attributes, its children are the caption."""


def figure_name(argument):
    name = directives.unchanged_required(argument).strip()
    if name not in FIGURES:
        raise ValueError(f'unknown figure "{name}", the figures are: {", ".join(sorted(FIGURES))}')
    return name


class FigureDirective(SphinxDirective):
    """What the plate and the diagram share: draw the figure, parse the text."""

    has_content = True
    option_spec = {
        "figure": figure_name,
        "svg": directives.unchanged_required,
        "label": directives.unchanged_required,
        "class": directives.class_option,
    }
    # Set by each directive: the node it builds and the name used in ids and classes.
    node_class = nodes.Element
    kind = ""
    # The width of a hand-written drawing, the width of the generated ones.
    width = 0

    def run(self):
        raise NotImplementedError

    def drawing(self, panels=None):
        if ("figure" in self.options) == ("svg" in self.options):
            raise self.error(f'the {self.name} directive takes one of ":figure:" and ":svg:"')
        # Several figures share a page: their SVG ids start with the kind and the
        # rank of the figure in the page, "plate-2-r" for a pattern of the second plate.
        prefix = f"{self.kind}-{self.env.new_serialno(f'sm-{self.kind}') + 1}"
        if "svg" in self.options:
            return prefix, self.hand_drawing(prefix, panels)
        if panels is not None:
            raise self.error('":panels:" goes with ":svg:"; a figure function gives its panels')
        # A change to the drawing code rebuilds the pages that show a figure.
        for source in FIGURE_SOURCES:
            self.env.note_dependency(source)
        try:
            return prefix, draw(self.options["figure"], prefix)
        except ValueError as error:
            raise self.error(f'figure "{self.options["figure"]}": {error}') from error

    def hand_drawing(self, prefix, panels):
        """The drawing of the ``:svg:`` file, checked by ``static_svg``."""
        name, path = self.env.relfn2path(self.options["svg"], self.env.docname)
        # The file, and the theme variables it may use, rebuild the page.
        self.env.note_dependency(path)
        self.env.note_dependency(str(THEME_CSS))
        try:
            return load(path, name, f"{prefix}-svg", self.width, panels)
        except ValueError as error:
            raise self.error(str(error)) from error

    def make_node(self, prefix, drawing, **attributes):
        source = self.options.get("figure") or self.options["svg"]
        node = self.node_class(figure=source, **attributes)
        node["classes"] += [f"sm-{self.kind}", *self.options.get("class", [])]
        self.set_source_info(node)
        node["svg"] = drawing.svg(f"{prefix}-art")
        node["art_id"] = f"{prefix}-art"
        if "label" in self.options:
            node["label"] = self.options["label"]
            node["ids"].append(nodes.make_id(self.options["label"]))
        else:
            node["ids"].append(prefix)
        self.state.nested_parse(self.content, self.content_offset, node)
        return node


class PlateDirective(FigureDirective):
    required_arguments = 1
    final_argument_whitespace = True
    option_spec = {
        **FigureDirective.option_spec,
        "columns": lambda value: directives.choice(value, ("1", "2")),
        "panels": directives.unchanged_required,
    }
    node_class = plate
    kind = "plate"
    width = PLATE_WIDTH

    def run(self):
        prefix, drawing = self.drawing(self.options.get("panels"))
        node = self.make_node(
            prefix, drawing, title=self.arguments[0], columns=int(self.options.get("columns", "2"))
        )
        node["panels"] = [list(panel) for panel in drawing.panels]
        if drawing.panels:
            node["classes"].append("sm-plate-stacked")
        for paragraph in node.findall(nodes.paragraph):
            # MyST may open a paragraph with an empty text node
            children = [
                c for c in paragraph.children if not (isinstance(c, nodes.Text) and not c.strip())
            ]
            first = children[0] if children else None
            if isinstance(first, nodes.strong) and FIG_LABEL.fullmatch(first.astext().rstrip(".")):
                first["classes"].append("sm-plate-fig")
            if len(children) == 1 and isinstance(first, nodes.emphasis):
                paragraph["classes"].append("sm-plate-note")
        for emphasis in node.findall(nodes.emphasis):
            if re.fullmatch(r"[a-z]", emphasis.astext()):
                emphasis["classes"].append("sm-callout")
        return [node]


class DiagramDirective(FigureDirective):
    node_class = diagram
    kind = "diagram"
    width = DIAGRAM_WIDTH

    def run(self):
        prefix, drawing = self.drawing()
        if "svg" in self.options and drawing.panels:
            raise self.error(
                f'"{self.options["svg"]}" has data-panels, but a diagram is never stacked'
            )
        return [self.make_node(prefix, drawing)]


class NumberPlates(SphinxTransform):
    """Number the plates of a page in document order and register the labels."""

    default_priority = 500

    def apply(self, **kwargs):
        std = self.env.domains["std"]
        for number, node in enumerate(self.document.findall(plate), start=1):
            node["number"] = number
            if "label" in node:
                std.labels[node["label"]] = (
                    self.env.docname,
                    node["ids"][0],
                    f"Plate {number} ({node['title']})",
                )
                std.anonlabels[node["label"]] = (self.env.docname, node["ids"][0])
        for node in self.document.findall(diagram):
            if "label" in node:
                std.anonlabels[node["label"]] = (self.env.docname, node["ids"][0])


def visit_plate_html(self, node):
    n = node["number"]
    panels = node["panels"]
    self.body.append(self.starttag(node, "figure"))
    self.body.append('<div class="sm-plate-frame"><div class="sm-plate-inner">')
    self.body.append(
        f'<div class="sm-plate-head"><span class="sm-plate-title">{escape(node["title"])}</span>'
        f'<span class="sm-plate-number">Plate {n}</span></div>'
    )
    if panels:
        self.body.append(f'<div class="sm-plate-art">{node["svg"]}</div>')
        # On narrow screens each panel shows one figure of the drawing through its
        # own viewBox and a <use> of the whole drawing, so nothing is drawn twice.
        # The panels are hidden from screen readers: the drawing describes itself.
        self.body.append('<div class="sm-plate-stack" aria-hidden="true">')
        for x, y, w, h, kind in panels:
            cls = f"sm-plate-panel sm-plate-panel-{kind}" if kind else "sm-plate-panel"
            self.body.append(
                f'<div class="{cls}"><svg viewBox="{x} {y} {w} {h}" width="{w}" height="{h}">'
                f'<use href="#{node["art_id"]}"/></svg></div>'
            )
        self.body.append("</div>")
    else:
        # Without panels, the drawing scrolls sideways inside the frame on narrow screens.
        self.body.append(
            '<p class="sm-plate-hint" aria-hidden="true">Scroll sideways to see the whole plate</p>'
        )
        self.body.append(
            '<div class="sm-plate-art sm-plate-scroll" tabindex="0" role="region"'
            f' aria-label="Plate {n}">{node["svg"]}</div>'
        )
    self.body.append("</div></div>\n")
    one = " sm-plate-legend-one" if node["columns"] == 1 else ""
    self.body.append(
        '<figcaption class="sm-plate-legend"><p class="sm-plate-legend-title">Explanation of Plate'
        f' {n}</p><div class="sm-plate-legend-cols{one}">\n'
    )


def depart_plate_html(self, _node):
    self.body.append("</div></figcaption></figure>\n")


def visit_diagram_html(self, node):
    self.body.append(self.starttag(node, "figure"))
    # On narrow screens the drawing keeps a legible size and scrolls sideways.
    self.body.append(
        '<p class="sm-diagram-hint" aria-hidden="true">Scroll sideways to see the whole figure</p>'
    )
    self.body.append(
        '<div class="sm-diagram-art" tabindex="0" role="region"'
        f' aria-label="Figure">{node["svg"]}</div>'
    )
    if node.children:
        self.body.append("<figcaption>")


def depart_diagram_html(self, node):
    if node.children:
        self.body.append("</figcaption>")
    self.body.append("</figure>\n")


def visit_text_only(_self, _node):
    pass


def depart_text_only(_self, _node):
    pass


def setup(app):
    others = (visit_text_only, depart_text_only)
    for node, html in (
        (plate, (visit_plate_html, depart_plate_html)),
        (diagram, (visit_diagram_html, depart_diagram_html)),
    ):
        app.add_node(node, html=html, latex=others, text=others, man=others, texinfo=others)
    app.add_directive("plate", PlateDirective)
    app.add_directive("diagram", DiagramDirective)
    app.add_transform(NumberPlates)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
