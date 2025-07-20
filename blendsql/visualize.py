from sqlglot import exp
import graphviz
from typing import Optional

from blendsql.parse.dialect import KeywordArgument

FONT = "Sans Serif,monospace"
BASE_FONTSIZE = 12
SMALL_FONTSIZE = 8

DEFAULT_NODE_COLOR = "#0044ff22"
COLOR_MAP = {
    "Select": "lightgreen",
    "From": "#ff880022",
    "Where": "lightyellow",
    "Join": "#00008844",
    "Column": "lightcyan",
    "Table": "lightgray",
    "Literal": "white",
    "Identifier": "lavender",
    "Binary": "wheat",
    "Function": "lightsteelblue",
    "Order": "lightseagreen",
    "Group": "lightgoldenrodyellow",
    "Having": "lightpink",
    "Union": "lightsalmon",
    "With": "lightblue",
    "BlendSQLFunction": "#f78e69",
    "Boolean": "white",
    "KeywordArgument": "white",
}


class SQLGlotASTVisualizer:
    def __init__(self):
        self.dot = None
        self.node_counter = 0

    def create_graph(self, name: str = "sqlglot_ast") -> graphviz.Digraph:
        """Create a new Graphviz directed graph with styling."""
        dot = graphviz.Digraph(name, comment="SQLGlot AST")

        # Set graph attributes for better PDF output
        dot.attr(rankdir="TB")
        dot.attr(
            "node",
            shape="box",
            style="rounded,filled",
            fillcolor="lightblue",
            fontname=FONT,
        )
        dot.attr("edge", fontname=FONT)
        dot.attr("graph", dpi="300")  # High DPI for better PDF quality

        return dot

    def get_node_label(self, node: exp.Expression) -> str:
        """Generate a descriptive label for an AST node."""
        node_type = type(node).__name__
        text_with_size = lambda s, size: f"<FONT POINT-SIZE='{size}'>{s}</FONT>"

        def format_label(header: str, subheader: Optional[str] = None) -> str:
            if subheader:
                return f"<<b>{text_with_size(header, BASE_FONTSIZE)}</b><BR/>{text_with_size(subheader, SMALL_FONTSIZE)}>"
            return f"<<b>{text_with_size(header, BASE_FONTSIZE)}</b>>"

        if isinstance(node, (exp.From, exp.EQ, exp.Count)):
            return node_type
        if isinstance(node, (exp.Column, exp.Table, exp.TableAlias)):
            # Get Identifier
            return format_label(node.this.this, node_type)
        elif isinstance(node, KeywordArgument):
            if not isinstance(node.value, (exp.Literal, exp.Boolean)):
                return format_label(
                    f"{node.name}={node.value}", type(node.value).__name__
                )
            return format_label(f"{node.name}={node.value}")
        elif isinstance(node, exp.Predicate):
            return node_type
        elif isinstance(node, exp.Star):
            return "*"
        elif isinstance(node, exp.Boolean):
            return format_label(node.to_py())
        elif isinstance(node, exp.Literal):
            return format_label(node.to_py())
        elif hasattr(node, "this") and node.this:
            if isinstance(node.this, str) and len(node.this) > 1:
                return format_label(node.this, node_type)
            elif hasattr(node.this, "name") and len(node.this.name) > 1:
                return format_label(node.this.name)
        return node_type

    def get_node_color(self, node: exp.Expression) -> str:
        """Get color based on node type."""
        node_type = type(node).__name__

        return COLOR_MAP.get(node_type, DEFAULT_NODE_COLOR)

    def add_node_to_graph(
        self, node: exp.Expression, parent_id: str = None, edge_label: str = None
    ) -> str:
        """Add a node and its children to the graph recursively."""
        node_id = f"node_{self.node_counter}"
        self.node_counter += 1

        label = self.get_node_label(node)
        color = self.get_node_color(node)

        self.dot.node(node_id, label=label, fillcolor=color)

        if parent_id:
            edge_attrs = {}
            if edge_label:
                edge_attrs["label"] = edge_label
            self.dot.edge(parent_id, node_id, **edge_attrs)
        if not isinstance(node, KeywordArgument):
            if hasattr(node, "args") and node.args:
                for _arg_name, arg_value in node.args.items():
                    if arg_value is None:
                        continue
                    if isinstance(arg_value, (exp.Identifier,)):  # Don't expand these
                        continue
                    if isinstance(arg_value, exp.Expression):
                        self.add_node_to_graph(arg_value, node_id)
                    elif isinstance(arg_value, list):
                        for _i, item in enumerate(arg_value):
                            if isinstance(item, exp.Expression):
                                # label = f"{arg_name}[{i}]" if len(arg_value) > 1 else arg_name
                                self.add_node_to_graph(item, node_id)
                    elif isinstance(arg_value, dict):
                        for _key, value in arg_value.items():
                            if isinstance(value, exp.Expression):
                                self.add_node_to_graph(value, node_id)

        return node_id

    def visualize(self, node: exp.Expression) -> graphviz.Digraph:
        """Parse SQL and create visualization."""
        self.node_counter = 0
        self.dot = self.create_graph()

        self.add_node_to_graph(node)

        return self.dot


if __name__ == "__main__":
    from blendsql.parse.dialect import BlendSQLSQLite
    from sqlglot import parse_one

    visualizer = SQLGlotASTVisualizer()

    query = """
    SELECT * FROM People P
    WHERE P.Name IN {{
        LLMQA('First 3 presidents of the U.S?', quantifier='{3}')
    }}
    """
    # Generate visualization
    dot = visualizer.visualize(parse_one(query, dialect=BlendSQLSQLite))

    # Save as PDF
    dot.render("blendsql_diagram.pdf", format="pdf", cleanup=True)
