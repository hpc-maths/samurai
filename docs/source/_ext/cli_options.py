"""Mark inline code that holds command-line options, so a line never splits one.

A browser may break a line after any hyphen, so `--with-corner` can end a line with
"--" and start the next with "with-corner". No CSS property removes that break
opportunity while keeping the emergency wrap that long paths and type names need.
This transform tags every inline literal whose text starts with "--" with the class
``sm-cli``; samurai.css keeps each word of such a literal on one line. Options are
short, so they always fit the column, and lines still break at the spaces between
words, as in `--level 8 --Tf 0.5`.
"""

from docutils import nodes
from sphinx.transforms import SphinxTransform

CLASS = "sm-cli"


class MarkCliOptions(SphinxTransform):
    """Add the ``sm-cli`` class to each inline literal that starts with "--"."""

    default_priority = 500

    def apply(self, **kwargs):
        """Tag the matching literals of the document."""
        for literal in self.document.findall(nodes.literal):
            if literal.astext().startswith("--") and CLASS not in literal["classes"]:
                literal["classes"].append(CLASS)


def setup(app):
    """Register the transform."""
    app.add_transform(MarkCliOptions)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
