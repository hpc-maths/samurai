"""Mark short hyphenated inline code, so a line never splits it at a hyphen.

A browser may break a line after any hyphen, so `--with-corner` can end a line with
"--" and `y-offset[2]` can end one with "y-". No CSS property removes that break
opportunity while keeping the emergency wrap that long paths and type names need.

This transform tags an inline literal when one of its words holds a hyphen and the
longest word fits the column. A word is a run of non-space characters, which the HTML
writer puts in its own ``span.pre``; samurai.css keeps each of those spans on one line,
so lines still break at the spaces between words, as in `--level 8 --Tf 0.5`. The
first and last words also count the punctuation glued to the literal, as in
"(`pkg-config`)", since no line may start with ")".

The limits come from the narrowest column that holds running text, a list item:

- ``sm-hyphenated``: at most ``NARROW`` characters, kept whole at every width. On a
  390 px phone a list item is 338 px wide and inline code advances 9 px per character,
  so 37 characters fit.
- ``sm-hyphenated-wide``: at most ``WIDE`` characters, kept whole from a 48em (768 px)
  viewport on, where the article column reaches its full 736 px, a list item 716 px,
  and inline code advances 10 px per character, so 71 characters fit. Below that width
  it keeps its hyphen breaks.

A literal with a longer word is left alone: it keeps every break opportunity rather
than spill out of its column or push a ")" onto a line of its own.
"""

import re

from docutils import nodes
from sphinx.transforms import SphinxTransform

NARROW = 37
WIDE = 71

OPENING = re.compile(r"[(\[{\"']*$")
CLOSING = re.compile(r"^[)\]}\"'.,;:!?]*")


def _neighbour_text(node, step):
    """Return the text right before (``step=-1``) or after (``step=1``) an inline node.

    The search climbs out of inline ancestors such as a reference, whose text is glued
    to the literal, but stops at the enclosing paragraph or other block.
    """
    while node.parent is not None:
        parent = node.parent
        index = parent.index(node) + step
        if 0 <= index < len(parent):
            sibling = parent[index]
            return sibling.astext() if isinstance(sibling, nodes.Text) else ""
        if not isinstance(parent, nodes.Inline):
            break
        node = parent
    return ""


def hyphen_class(literal):
    """Return the class that keeps this inline literal whole, or None."""
    words = literal.astext().split()
    if not any("-" in word for word in words):
        return None
    widths = [len(word) for word in words]
    widths[0] += len(OPENING.search(_neighbour_text(literal, -1)).group())
    widths[-1] += len(CLOSING.match(_neighbour_text(literal, 1)).group())
    if max(widths) <= NARROW:
        return "sm-hyphenated"
    if max(widths) <= WIDE:
        return "sm-hyphenated-wide"
    return None


class MarkHyphenatedCode(SphinxTransform):
    """Tag each hyphenated inline literal that fits the column with its class."""

    default_priority = 500

    def apply(self, **kwargs):
        """Tag the matching literals of the document."""
        for literal in self.document.findall(nodes.literal):
            name = hyphen_class(literal)
            if name and name not in literal["classes"]:
                literal["classes"].append(name)


def setup(app):
    """Register the transform."""
    app.add_transform(MarkHyphenatedCode)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
