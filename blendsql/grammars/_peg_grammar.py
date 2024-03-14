import pyparsing
import pyparsing as pp

pp.ParserElement.enable_packrat()

"""Defines PEG (parsing expression grammar) in pyparsing 
grammar to extract positional + named args from function.

Since we use sqlglot for the main SQLite parsing, we only need to define the BlendSQL
operations (between "{{", "}}") below.
"""
grammar = pp.Forward()
blendsql_function_name = pp.Word(pp.alphanums + "_")
nums = pp.Word(pp.nums)

# Parse action to remove opening and closing quotes
single_qs = pp.QuotedString(quoteChar='"', multiline=True, escChar="\\")
double_qs = pp.QuotedString(quoteChar="'", multiline=True, escChar="\\")
str_arg = single_qs | double_qs
# Below is modified from pp.dbl_slash_comment
# TODO: confirm below is working
sql_style_comment = pp.Regex(r"--(?:\\\n|[^\n])*").set_name("-- comment")
"Comment of the form ``-- ... (to end of line)``"


def flatten_with_parens(xss):
    """hacky. Since nested_expr removes parentheses,
    we add them back here.
    """
    for idx in range(len(xss)):
        xss[idx].append(")")
        xss[idx].insert(0, "(")
    return [x for xs in xss for x in xs]


blendsql_arg = pp.nested_expr(
    "(", ")", ignore_expr=sql_style_comment | pyparsing.quoted_string
).setParseAction(flatten_with_parens, " ".join)
int_arg = nums.setParseAction(lambda x: int(x[0]))
float_arg = pp.Combine(pp.Optional("-") + nums + "." + nums).setParseAction(
    lambda x: float(x[0])
)
arg = str_arg | float_arg | int_arg | blendsql_arg
positional_command_arg = arg + ~pp.FollowedBy(pp.Char("=")) | pp.Suppress(
    ","
) + arg + ~pp.FollowedBy(pp.Char("="))
named_command_arg = pp.Group(
    pp.Word(pp.alphas + "_") + pp.Char("=") + arg
    | pp.Suppress(",") + pp.Word(pp.alphas + "_") + pp.Char("=") + arg
)

function_call_start = pp.Suppress(pp.Literal("{{"))
function_call_end = pp.Suppress(pp.Literal("}}"))
grammar <<= (
    function_call_start
    + blendsql_function_name("function")
    + pp.Literal("(")
    + pp.ZeroOrMore(positional_command_arg)("args")
    + pp.ZeroOrMore(named_command_arg)("kwargs")
    + pp.Literal(")")
    + function_call_end
)

if __name__ == "__main__":
    pp.autoname_elements()
    grammar.create_diagram("blendsql_grammar.html", show_results_names=True)
