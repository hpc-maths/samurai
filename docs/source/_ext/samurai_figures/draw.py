"""Drawing primitives shared by the diagrams and the plates.

Every function returns a fragment of SVG markup as a string. Colours are CSS
variables of the theme (``_static/css/samurai.css``), so the drawings follow the
light and the dark themes; faces are CSS classes (``sm-fig-label`` sans,
``sm-fig-math`` serif italic, ``sm-fig-code`` mono, ``sm-fig-sc`` small capitals);
strokes use the three weights of the page rules. One drawing unit is one CSS pixel
when the drawing is shown at its natural width.
"""

from decimal import ROUND_HALF_UP, Decimal
from html import escape

INK = "var(--sm-ink)"
INK2 = "var(--sm-ink-2)"
INK3 = "var(--sm-ink-3)"
RED = "var(--sm-red)"
FIG = "var(--sm-fig-line)"
PAPER = "var(--sm-paper)"
WASH = "var(--sm-ink-wash)"
RWASH = "var(--sm-red-wash)"

# The hairline, the rule and the heavy rule of the page, and the faint outline of
# an empty cell.
HAIR, RULE, HEAVY, EMPTY = 0.5, 1, 1.5, 0.75


def num(v):
    """A coordinate with at most two decimals and no trailing zero, a half rounded up."""
    s = f"{Decimal(float(v)).quantize(Decimal('0.01'), ROUND_HALF_UP)}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


def _dash(dash):
    return f' stroke-dasharray="{dash}"' if dash else ""


def _extra(extra):
    return f" {extra}" if extra else ""


def text(x, y, s, anchor="start", size=11.5, fill=INK2, cls="sm-fig-label", knock=False, extra=""):
    """A text at (x, y). ``s`` is SVG markup: it may hold ``<tspan>`` elements.

    ``knock`` draws a halo of the paper colour behind the glyphs, so that the text
    stays legible over lines.
    """
    cls += " sm-knock" if knock else ""
    return (
        f'<text x="{num(x)}" y="{num(y)}" text-anchor="{anchor}" font-size="{size}" '
        f'fill="{fill}" class="{cls}"{_extra(extra)}>{s}</text>'
    )


def rect(x, y, w, h, fill="none", stroke=INK, sw=RULE, dash=""):
    return (
        f'<rect x="{num(x)}" y="{num(y)}" width="{num(w)}" height="{num(h)}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{_dash(dash)}/>'
    )


def line(x1, y1, x2, y2, stroke=INK, sw=RULE, dash="", extra=""):
    return (
        f'<line x1="{num(x1)}" y1="{num(y1)}" x2="{num(x2)}" y2="{num(y2)}" '
        f'stroke="{stroke}" stroke-width="{sw}"{_dash(dash)}{_extra(extra)}/>'
    )


def circle(x, y, r, fill="none", stroke="", sw=RULE, dash=""):
    outline = f' stroke="{stroke}" stroke-width="{sw}"{_dash(dash)}' if stroke else ""
    return f'<circle cx="{num(x)}" cy="{num(y)}" r="{num(r)}" fill="{fill}"{outline}/>'


def path(d, stroke=INK2, sw=RULE, extra=""):
    return f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}"{_extra(extra)}/>'


def bracket(xa, xb, y, label, color=INK, size=11.5, r=3, label_dy=15):
    """The half-open interval [a, b) under a run of cells.

    samurai intervals are half-open: the start belongs to the interval, the end does
    not. The segment starts with a filled dot and ends with an open dot.
    """
    s = line(xa, y, xb, y, stroke=color)
    s += circle(xa, y, r, fill=color)
    s += circle(xb, y, r, fill=PAPER, stroke=color)
    if label:
        # a paper ground the width of the mono label, so that guides do not cross it
        w = 0.6 * size * len(label) + 4
        s += rect(
            (xa + xb - w) / 2, y + label_dy - 0.85 * size, w, size + 2, fill=PAPER, stroke="none"
        )
        s += text(
            (xa + xb) / 2,
            y + label_dy,
            label,
            anchor="middle",
            size=size,
            fill=color,
            cls="sm-fig-code",
            knock=True,
        )
    return s


def cell_row(x0, y, cw, to, cells, ch=26, start=0, tone="ink", brackets=True, label_size=11.5):
    """A row of cells ``start`` to ``to - 1`` on a number line.

    The cells outside the set are drawn faint, the cells of the set filled: in ink
    for an operand (``tone="ink"``), in red for a result (``tone="red"``). ``cells``
    is a list of half-open intervals (a, b).
    """
    g = ""
    for k in range(start, to):
        if not any(a <= k < b for a, b in cells):
            g += rect(x0 + (k - start) * cw, y, cw, ch, stroke=FIG, sw=EMPTY)
    fill, stroke = (RWASH, RED) if tone == "red" else (WASH, INK)
    for a, b in cells:
        for k in range(a, b):
            g += rect(x0 + (k - start) * cw, y, cw, ch, fill=fill, stroke=stroke)
        if brackets:
            g += bracket(
                x0 + (a - start) * cw,
                x0 + (b - start) * cw,
                y + ch + 9,
                f"[{a}, {b})",
                color=stroke,
                size=label_size,
            )
    return g


def arrow(id_, color):
    """An open arrow head, for ``marker-end``."""
    return (
        f'<marker id="{id_}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7"'
        ' markerHeight="7" orient="auto-start-reverse"><path d="M1,1.5 L9,5 L1,8.5" fill="none"'
        f' stroke="{color}" stroke-width="1.4"/></marker>'
    )


def patterns(p):
    """The fills and arrow heads of a plate, their ids prefixed with ``p``.

    ``{p}-r``: the cells a figure points out, the red wash crossed by fine red
    hatching at 45 degrees. ``{p}-k``: a region where the solution is constant,
    fine horizontal ink rules. ``{p}-a`` and ``{p}-ar``: arrow heads in ink and in red.
    """
    return (
        f'<pattern id="{p}-r" patternUnits="userSpaceOnUse" width="3.4" height="3.4"'
        f' patternTransform="rotate(45)"><rect width="3.4" height="3.4" fill="{RWASH}"/><line'
        f' x1="0" y1="0" x2="0" y2="3.4" stroke="{RED}" stroke-width="0.8"/></pattern><pattern'
        f' id="{p}-k" patternUnits="userSpaceOnUse" width="3" height="3"><line x1="0" y1="1.5"'
        f' x2="3" y2="1.5" stroke="{INK3}" stroke-width="{HAIR}"/></pattern>'
        + arrow(f"{p}-a", INK2)
        + arrow(f"{p}-ar", RED)
    )


def callout(x, y, tx, ty, letter, red=False):
    """A lettered callout: a dot on the target, a fine leader, an italic letter.

    The legend of the plate explains the letter.
    """
    return (
        line(x, y, tx, ty, stroke=INK, sw=HAIR)
        + circle(x, y, 1.8, fill=RED if red else INK)
        + text(
            tx, ty + 5, letter, anchor="middle", size=15, fill=INK, cls="sm-fig-math", knock=True
        )
    )


def fig_label(x, y, n, extra=""):
    """The "Fig. n." under a figure of a plate, the legend refers to it."""
    s = f"Fig. {n}." + (f' <tspan class="sm-fig-math">{extra}</tspan>' if extra else "")
    return text(x, y, s, anchor="middle", size=14, fill=INK, cls="sm-fig-sc")


def svg(width, height, label, defs, body, id_=""):
    """The whole drawing. ``label`` describes it to screen readers."""
    ident = f' id="{id_}"' if id_ else ""
    return (
        f'<svg{ident} class="sm-fig" viewBox="0 0 {num(width)} {num(height)}" width="{num(width)}"'
        f' height="{num(height)}" role="img"'
        f' aria-label="{escape(label)}"><defs>{defs}</defs>{body}</svg>'
    )
