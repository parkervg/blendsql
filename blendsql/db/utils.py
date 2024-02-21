import re


def single_quote_escape(s):
    return re.sub(r"(?<=[^'])'(?=[^'])", "''", s)


def double_quote_escape(s):
    return re.sub(r'(?<=[^"])"(?=[^"])', '""', s)


def escape(s):
    return single_quote_escape(double_quote_escape(s))
