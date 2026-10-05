"""Numbered definitions and remarks, set the way a textbook sets them.

Two directives, written in MyST as fenced blocks::

    ```{definition} level
    :label: def-level

    A cell at level $\\ell$ has a side of $2^{-\\ell}$.
    ```

    ```{remark}
    This property is needed for mesh adaptation.
    ```

Each statement starts with its label, "Definition 1." or "Remark 2.", in small
capitals, followed by the defined term in italics when one is given. The
definitions and remarks of a page share one sequence of numbers, so a reader can
find "Remark 5" by counting. ``:label:`` makes the statement a target for
``{ref}`label```, which prints "Definition 1 (level)".
"""

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.transforms import SphinxTransform
from sphinx.util.docutils import SphinxDirective

KINDS = {"definition": "Definition", "remark": "Remark"}


class statement(nodes.General, nodes.Element):
    """A definition or a remark: a block whose first paragraph opens with its label."""


class statement_label(nodes.Inline, nodes.TextElement):
    """The "Definition 1." at the start of a statement, filled in by NumberStatements."""


class StatementDirective(SphinxDirective):
    has_content = True
    optional_arguments = 1
    final_argument_whitespace = True
    option_spec = {"label": directives.unchanged_required, "class": directives.class_option}

    def run(self):
        kind = self.name
        term = self.arguments[0] if self.arguments else ""
        node = statement(
            kind=kind,
            term=term,
            classes=["sm-statement", f"sm-{kind}", *self.options.get("class", [])],
        )
        self.set_source_info(node)
        if "label" in self.options:
            node["label"] = self.options["label"]
            node["ids"].append(nodes.make_id(self.options["label"]))
        self.state.nested_parse(self.content, self.content_offset, node)

        opening = [statement_label("", "", classes=["sm-statement-label"])]
        if term:
            opening += [nodes.Text(" "), nodes.emphasis(term, term, classes=["sm-statement-term"])]
        opening.append(nodes.Text(" "))
        first = node.children[0] if node.children else None
        if isinstance(first, nodes.paragraph):
            first[0:0] = opening
        else:
            node.insert(0, nodes.paragraph("", "", *opening))
        return [node]


class NumberStatements(SphinxTransform):
    """Number the statements of a page in document order and register their labels."""

    default_priority = 500

    def apply(self, **kwargs):
        std = self.env.domains["std"]
        for number, node in enumerate(self.document.findall(statement), start=1):
            name = KINDS[node["kind"]]
            node["number"] = number
            for label in node.findall(statement_label):
                label += nodes.Text(f"{name} {number}.")
                break
            if "label" in node:
                title = f"{name} {number}" + (f" ({node['term']})" if node["term"] else "")
                std.labels[node["label"]] = (self.env.docname, node["ids"][0], title)
                std.anonlabels[node["label"]] = (self.env.docname, node["ids"][0])


def visit_statement_html(self, node):
    self.body.append(self.starttag(node, "div"))


def depart_statement_html(self, _node):
    self.body.append("</div>\n")


def visit_statement_label_html(self, node):
    self.body.append(self.starttag(node, "span", ""))


def depart_statement_label_html(self, _node):
    self.body.append("</span>")


def visit_statement_other(_self, _node):
    pass


def depart_statement_other(_self, _node):
    pass


def setup(app):
    app.add_node(
        statement,
        html=(visit_statement_html, depart_statement_html),
        latex=(visit_statement_other, depart_statement_other),
        text=(visit_statement_other, depart_statement_other),
        man=(visit_statement_other, depart_statement_other),
        texinfo=(visit_statement_other, depart_statement_other),
    )
    app.add_node(
        statement_label,
        html=(visit_statement_label_html, depart_statement_label_html),
        latex=(visit_statement_other, depart_statement_other),
        text=(visit_statement_other, depart_statement_other),
        man=(visit_statement_other, depart_statement_other),
        texinfo=(visit_statement_other, depart_statement_other),
    )
    for kind in KINDS:
        app.add_directive(kind, StatementDirective)
    app.add_transform(NumberStatements)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
