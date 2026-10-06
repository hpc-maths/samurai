"""Hand-written SVG figures, for the schematics that depend on no program data.

The ``plate`` and ``diagram`` directives take ``:svg: <file>`` instead of
``:figure: <function>``. ``parse`` checks the file and turns it into the same
``Drawing`` a figure function returns, so a hand-written figure is inlined, themed
and stacked like a generated one.

The file must follow the rules of the generated figures, and the build fails on the
first rule it breaks:

- the root ``<svg>`` has ``viewBox="0 0 W H"``, W being the width of a plate
  (``PLATE_WIDTH``) or of a diagram (``DIAGRAM_WIDTH``), so that one unit is one CSS
  pixel and the strokes keep their weights, and an ``aria-label`` that describes it;
- every ``fill``, ``stroke``, ``stop-color`` and ``color`` (attribute, ``style``
  attribute or ``<style>`` rule) is ``none``, ``currentColor``, a ``var(--sm-*)``
  variable of ``samurai.css`` or a ``url(#id)`` of the file, never a fixed colour;
- the elements are drawing elements only: no ``<script>``, no ``<image>``, no
  ``<foreignObject>``, no link, filter or animation;
- every reference (``href``, ``url(...)``) points to an id of the file.

The ids of the file are prefixed, so several figures share a page, and the rules of
a ``<style>`` apply inside the drawing only. Shapes without a ``fill`` take the ink
colour of the theme instead of black.
"""

import re
from functools import cache
from html import escape
from pathlib import Path
from xml.parsers import expat

from .draw import num
from .registry import Drawing

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
XML_NS = "http://www.w3.org/XML/1998/namespace"

THEME_CSS = Path(__file__).resolve().parents[2] / "_static" / "css" / "samurai.css"

ELEMENTS = frozenset(
    (
        "g defs symbol use title desc style rect circle ellipse line polyline polygon path"
        " text tspan textPath marker pattern clipPath mask linearGradient radialGradient stop"
    ).split()
)
FORBIDDEN = {
    "script": "a <script> is not allowed: a figure is a static drawing",
    "image": "an <image> is not allowed: draw the figure in vector shapes, or show a"
    " raster image with the image directive",
    "foreignObject": "a <foreignObject> is not allowed: it embeds HTML outside the drawing",
}
COLOURS = frozenset(("fill", "stroke", "stop-color", "color", "flood-color", "lighting-color"))
# Attributes of the root <svg> that the directive replaces or reads.
ROOT = frozenset(("viewBox", "width", "height", "aria-label", "role", "version", "data-panels"))
ROOT_REFUSED = frozenset(("id", "x", "y", "preserveAspectRatio"))
ID_LISTS = frozenset(("aria-labelledby", "aria-describedby"))
KINDS = ("half", "wide")

URL = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)")
THEME_VAR = re.compile(r"var\(\s*--sm-([a-z0-9-]+)\s*\)")
NAME = re.compile(r"-?[A-Za-z_][\w-]*")
ID_SELECTOR = re.compile(r"#(-?[A-Za-z_][\w-]*)")
RULE = re.compile(r"([^{}]*)\{([^{}]*)\}")


class Node:
    """An element of the file: its local name, attributes, children and line."""

    def __init__(self, tag, attributes, line):
        """Hold ``tag``, its ``attributes`` and its ``line``, with no children yet."""
        self.tag, self.attributes, self.line = tag, attributes, line
        self.children = []


@cache
def theme_variables():
    """The names of the ``--sm-*`` variables that ``samurai.css`` defines."""
    return frozenset(re.findall(r"--sm-([a-z0-9-]+)\s*:", THEME_CSS.read_text(encoding="utf-8")))


class _Figure:
    """One file being checked and rewritten. Every error names the file and the line."""

    def __init__(self, name, prefix):
        self.name, self.prefix = name, prefix
        self.ids = set()

    def fail(self, node, message):
        where = f"{self.name}, line {node.line}, <{node.tag}>" if node else self.name
        raise ValueError(f"{where}: {message}")

    # -- reading ------------------------------------------------------------------

    def read(self, source):
        """The tree of the file, comments and processing instructions left out."""
        parser = expat.ParserCreate(namespace_separator=" ")
        stack, root = [], []

        def tag_of(qualified):
            ns, _, local = qualified.rpartition(" ")
            if ns not in ("", SVG_NS):
                self.fail(
                    None,
                    f"line {parser.CurrentLineNumber}: the element {local} of the namespace"
                    f" {ns} is not SVG; remove the editor metadata",
                )
            return local

        def start(qualified, attributes):
            node = Node(tag_of(qualified), {}, parser.CurrentLineNumber)
            for key, value in attributes.items():
                node.attributes[self.attribute_name(node, key)] = value
            (stack[-1].children if stack else root).append(node)
            stack.append(node)

        def text(data):
            if stack:
                stack[-1].children.append(data)

        def doctype(*_):
            self.fail(None, f"line {parser.CurrentLineNumber}: a DOCTYPE is not allowed")

        parser.StartElementHandler = start
        parser.EndElementHandler = lambda _: stack.pop()
        parser.CharacterDataHandler = text
        parser.StartDoctypeDeclHandler = doctype
        try:
            parser.Parse(source, True)
        except expat.ExpatError as error:
            self.fail(None, f"not well-formed XML: {error}")
        return root[0]

    def attribute_name(self, node, key):
        ns, _, local = key.rpartition(" ")
        if ns == "":
            return local
        if ns == XLINK_NS and local == "href":
            return "href"
        if ns == XML_NS:
            return f"xml:{local}"
        return self.fail(
            node,
            f"the attribute {local} of the namespace {ns} is not SVG; remove the editor metadata",
        )

    # -- checking and rewriting ------------------------------------------------------

    def collect_ids(self, node):
        if "id" in node.attributes:
            ident = node.attributes["id"]
            if not NAME.fullmatch(ident):
                self.fail(node, f'id "{ident}" is not a valid id')
            if ident in self.ids:
                self.fail(node, f'id "{ident}" is used twice')
            self.ids.add(ident)
        for child in node.children:
            if isinstance(child, Node):
                self.collect_ids(child)

    def ref(self, node, ident, where):
        if ident not in self.ids:
            self.fail(node, f'{where} points to "#{ident}", which is no id of the file')
        return f"{self.prefix}-{ident}"

    def urls(self, node, value, where):
        """``value`` with the ``url(#id)`` it holds prefixed."""

        def one(match):
            target = match.group(2)
            if not target.startswith("#"):
                self.fail(node, f'{where} "{value}" is an external reference, url({target})')
            return f"url(#{self.ref(node, target[1:], where)})"

        return URL.sub(one, value)

    def colour(self, node, prop, value, where):
        """Fail unless ``value`` of the colour property ``prop`` follows the theme."""
        value = value.strip()
        plain = re.sub(r"\s*!important$", "", value)
        if plain == "none" or plain.lower() == "currentcolor":
            return
        if URL.fullmatch(plain):
            return
        match = THEME_VAR.fullmatch(plain)
        if match and match.group(1) in theme_variables():
            return
        if match:
            self.fail(
                node,
                f'{where} {prop} "{value}": samurai.css defines no variable --sm-{match.group(1)}',
            )
        self.fail(
            node,
            f'{where} {prop} "{value}" is a fixed colour; use none, currentColor or a'
            " var(--sm-*) variable of samurai.css, so that the figure follows the theme",
        )

    def declarations(self, node, block, where):
        """The CSS declarations of ``block``, checked and with their ``url()`` prefixed."""
        out = []
        for declaration in block.split(";"):
            if not declaration.strip():
                continue
            prop, colon, value = declaration.partition(":")
            prop = prop.strip().lower()
            if not colon:
                self.fail(node, f'{where} "{declaration.strip()}" is not a CSS declaration')
            if prop.startswith("--"):
                self.fail(
                    node,
                    f"{where} defines the custom property {prop}; use a variable" " of samurai.css",
                )
            if prop in COLOURS:
                self.colour(node, prop, value, where)
            out.append(f"{prop}:{self.urls(node, value.strip(), where)}")
        return ";".join(out)

    def stylesheet(self, node, css):
        """The rules of a ``<style>``, scoped to the drawing."""
        css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
        if "@" in css:
            self.fail(node, "an at-rule (@import, @media, ...) is not allowed in a <style>")
        rules = []
        for selectors, block in RULE.findall(css):
            scoped = []
            for selector in self.split_selectors(node, selectors):

                def ident(match, selector=selector):
                    return "#" + self.ref(node, match.group(1), f'the selector "{selector}"')

                scoped.append(f".{self.prefix} {ID_SELECTOR.sub(ident, selector)}")
            body = self.declarations(node, block, f'the rule "{selectors.strip()}":')
            rules.append(f"{','.join(scoped)}{{{body}}}")
        rest = RULE.sub("", css).strip()
        if rest:
            self.fail(node, f'"{rest[:40]}" is not a CSS rule')
        return "".join(rules)

    def split_selectors(self, node, selectors):
        parts, depth, start = [], 0, 0
        for i, c in enumerate(selectors):
            depth += (c == "(") - (c == ")")
            if c == "," and depth == 0:
                parts.append(selectors[start:i].strip())
                start = i + 1
        parts.append(selectors[start:].strip())
        if not all(parts):
            self.fail(node, f'the selector list "{selectors.strip()}" has an empty selector')
        return parts

    def element(self, node):
        """The markup of an element of the drawing."""
        if node.tag in FORBIDDEN:
            self.fail(node, FORBIDDEN[node.tag])
        if node.tag not in ELEMENTS:
            self.fail(
                node,
                f"the element <{node.tag}> is not allowed; a figure uses only"
                f" {', '.join(f'<{e}>' for e in sorted(ELEMENTS))}",
            )
        if node.tag == "stop" and not (
            "stop-color" in node.attributes or "stop-color" in node.attributes.get("style", "")
        ):
            self.fail(node, "a <stop> needs a stop-color, the default one is black")
        if node.tag == "style":
            if any(isinstance(c, Node) for c in node.children):
                self.fail(node, "a <style> holds CSS only, no element")
            return f"<style>{escape(self.stylesheet(node, ''.join(node.children)), False)}</style>"
        inner = "".join(
            self.element(c) if isinstance(c, Node) else escape(c, False) for c in node.children
        )
        return f"<{node.tag}{self.attributes(node, node.attributes)}>{inner}</{node.tag}>"

    def attributes(self, node, attributes):
        out = ""
        for key, value in attributes.items():
            where = f"the attribute {key}"
            if key.lower().startswith("on"):
                self.fail(node, f'the event handler {key}="{value}" is not allowed')
            if key == "id":
                value = f"{self.prefix}-{value}"
            elif key == "href":
                if not value.startswith("#"):
                    self.fail(
                        node,
                        f'href "{value}" is an external reference; point to an id' " of the file",
                    )
                value = "#" + self.ref(node, value[1:], "href")
            elif key in ID_LISTS:
                value = " ".join(self.ref(node, v, key) for v in value.split())
            elif key == "style":
                value = self.declarations(node, value, "the style attribute:")
            else:
                if key in COLOURS:
                    self.colour(node, key, value, "the attribute")
                value = self.urls(node, value, where)
            out += f' {key}="{escape(value)}"'
        return out

    # -- the drawing ---------------------------------------------------------------

    def size(self, root, width):
        box = root.attributes.get("viewBox", "").replace(",", " ").split()
        try:
            x, y, w, h = (float(v) for v in box)
        except ValueError:
            x = y = w = h = None
        if x != 0 or y != 0 or not h or h <= 0 or w != width:
            self.fail(
                root,
                f'viewBox "{root.attributes.get("viewBox", "")}" must be "0 0 {num(width)} H":'
                f" this figure is {num(width)} units wide, one unit for one CSS pixel",
            )
        for key, expected in (("width", w), ("height", h)):
            if key in root.attributes and root.attributes[key].removesuffix("px") != num(expected):
                self.fail(root, f'{key} "{root.attributes[key]}" differs from the viewBox')
        return w, h

    def panels(self, root, text, width, height):
        """The panels of ``text``: ``x y w h [kind]`` separated by semicolons."""
        out = []
        for item in text.split(";"):
            fields = item.split()
            if not fields:
                continue
            kind = fields.pop() if fields[-1] in KINDS else ""
            try:
                x, y, w, h = (float(v) for v in fields)
            except ValueError:
                x = y = w = h = None
            if x is None or w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
                self.fail(
                    root,
                    f'the panel "{item.strip()}" is not "x y w h" inside the drawing'
                    f" (0 0 {num(width)} {num(height)}), optionally followed by"
                    f" {' or '.join(KINDS)}",
                )
            out.append((num(x), num(y), num(w), num(h), kind))
        return tuple(out)

    def drawing(self, root, width, panels):
        if root.tag != "svg":
            self.fail(root, "the root element must be <svg>")
        w, h = self.size(root, width)
        label = root.attributes.get("aria-label", "").strip()
        if not label:
            self.fail(root, "the root <svg> needs an aria-label that describes the figure")
        if root.attributes.get("role", "img") != "img":
            self.fail(root, f'role "{root.attributes["role"]}" must be img')
        for key in ROOT_REFUSED & root.attributes.keys():
            self.fail(root, f"the root <svg> takes no {key}: the directive sets it")
        if panels is not None and "data-panels" in root.attributes:
            self.fail(root, "the panels are given twice, by data-panels and by :panels:")
        if panels is None:
            panels = root.attributes.get("data-panels", "")
        # The presentation attributes of the root go to a group around the drawing,
        # which scopes the <style> rules and gives shapes the ink of the theme.
        group = {"class": self.prefix, "fill": "var(--sm-ink)"}
        for key, value in root.attributes.items():
            if key not in ROOT:
                group[key] = f"{self.prefix} {value}" if key == "class" else value
        node = Node("g", group, root.line)
        node.children = root.children
        self.collect_ids(root)
        return Drawing(w, h, label, "", self.element(node), self.panels(root, panels, w, h))


def parse(source, name, prefix, width, panels=None):
    """The ``Drawing`` of the SVG markup ``source``, its ids starting with ``prefix``.

    ``name`` names the file in the errors, ``width`` is the width the drawing must
    have, ``panels`` overrides the ``data-panels`` attribute of the root. Raise
    ``ValueError`` when the markup breaks a rule of the figures.
    """
    figure = _Figure(name, prefix)
    return figure.drawing(figure.read(source), width, panels)


def load(path, name, prefix, width, panels=None):
    """The ``Drawing`` of the SVG file ``path``, see ``parse``."""
    try:
        source = Path(path).read_bytes()
    except OSError as error:
        raise ValueError(f"{name}: cannot read the file ({error.strerror})") from error
    return parse(source, name, prefix, width, panels)
