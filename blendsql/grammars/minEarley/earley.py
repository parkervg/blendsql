from collections import defaultdict, deque

from lark.lexer import Token
from lark.grammar import NonTerminal, Terminal
from lark.parse_tree_builder import ParseTreeBuilder


from .tree import Tree
from .earley_exceptions import UnexpectedCharacters, UnexpectedEOF
from .earley_forest import ForestSumVisitor, SymbolNode, TokenNode, ForestToParseTree
from .earley_analyzer import GrammarAnalyzer


class Item:
    def __init__(self, rule, ptr, start):
        self.rule = rule
        self.ptr = ptr
        self.start = start
        self.node = None

        self.is_complete = len(rule.expansion) == ptr

        if self.is_complete:
            self.s = rule.origin
            self.expect = None
            self.previous = (
                rule.expansion[ptr - 1] if ptr > 0 and len(rule.expansion) else None
            )
        else:
            self.s = (rule, ptr)
            self.expect = rule.expansion[ptr]
            self.previous = (
                rule.expansion[ptr - 1] if ptr > 0 and len(rule.expansion) else None
            )

        self._hash = hash((self.s, self.start))

    def advance(self):
        return Item(self.rule, self.ptr + 1, self.start)

    def __eq__(self, other):
        return self is other or (self.s == other.s and self.start == other.start)

    def __hash__(self):
        return self._hash

    def __repr__(self):
        before = (expansion.name for expansion in self.rule.expansion[: self.ptr])
        after = (expansion.name for expansion in self.rule.expansion[self.ptr :])
        symbol = f"{self.rule.origin.name} ::= {' '.join(before)}* {' '.join(after)}"
        return f"{symbol} ({self.start})"


class EarleyRegexpMatcher:
    def __init__(self, lexer_conf):
        self.regexps = {}
        for t in lexer_conf.terminals:
            regexp = t.pattern.to_regexp()
            if lexer_conf.use_bytes:
                regexp = regexp.encode("utf-8")

            self.regexps[t.name] = lexer_conf.re_module.compile(
                regexp, lexer_conf.g_regex_flags
            )

    def match(self, term, text, index=0):
        return self.regexps[term.name].match(text, index)


class Parser:
    def __init__(self, lexer_conf, parser_conf):
        self.lexer_conf = lexer_conf
        self.parser_conf = parser_conf
        self.term_matcher = EarleyRegexpMatcher(lexer_conf)

        self.TERMINALS = {
            sym for r in parser_conf.rules for sym in r.expansion if sym.is_term
        }
        self.NON_TERMINALS = {
            sym for r in parser_conf.rules for sym in r.expansion if not sym.is_term
        }

        self.predictions = {}
        analysis = GrammarAnalyzer(parser_conf)
        for rule in parser_conf.rules:
            if rule.origin not in self.predictions:
                self.predictions[rule.origin] = [
                    x.rule for x in analysis.expand_rule(rule.origin)
                ]

        self.complete_lex = False
        self.ignore = [Terminal(t) for t in lexer_conf.ignore]

    def parse(self, stream, start):
        start_symbol = NonTerminal(start)

        columns = [set()]
        to_scan = set()

        for rule in self.predictions[start_symbol]:
            item = Item(rule, 0, 0)
            if item.expect in self.TERMINALS:
                to_scan.add(item)
            else:
                columns[0].add(item)

        to_scan = self._parse(stream, columns, to_scan, start_symbol)

        solutions = [
            n.node
            for n in columns[-1]
            if n.is_complete
            and n.node is not None
            and n.s == start_symbol
            and n.start == 0
        ]
        if not solutions:
            expected_terminals = [t.expect.name for t in to_scan]
            raise UnexpectedEOF(
                stream, expected_terminals, state=frozenset(i.s for i in to_scan)
            )

        if len(solutions) > 1:
            raise AssertionError(
                "Earley should not generate multiple start symbol items!"
            )

        # Perform our SPPF -> AST conversion
        # TODO: check the code
        parse_tree_builder = ParseTreeBuilder(
            self.parser_conf.rules, Tree, False, True, True
        )
        _callbacks = parse_tree_builder.create_callback(None)
        transformer = ForestToParseTree(Tree, _callbacks, ForestSumVisitor(), True)
        return transformer.transform(solutions[0])

    def _parse(self, stream, columns, to_scan, start_symbol=None):
        def scan(i, to_scan):
            """The core Earley Scanner.

            This is a custom implementation of the scanner that uses the
            Lark lexer to match tokens. The scan list is built by the
            Earley predictor, based on the previously completed tokens.
            This ensures that at each phase of the parse we have a custom
            lexer context, allowing for more complex ambiguities."""

            node_cache = {}

            # 1) Loop the expectations and ask the lexer to match.
            # Since regexp is forward looking on the input stream, and we only
            # want to process tokens when we hit the point in the stream at which
            # they complete, we push all tokens into a buffer (delayed_matches), to
            # be held possibly for a later parse step when we reach the point in the
            # input stream at which they complete.
            for item in set(to_scan):
                m = match(item.expect, stream, i)
                if m:
                    t = Token(item.expect.name, m.group(0), i, text_line, text_column)
                    delayed_matches[m.end()].append((item, i, t))

                    if self.complete_lex:
                        s = m.group(0)
                        for j in range(1, len(s)):
                            m = match(item.expect, s[:-j])
                            if m:
                                t = Token(
                                    item.expect.name,
                                    m.group(0),
                                    i,
                                    text_line,
                                    text_column,
                                )
                                delayed_matches[i + m.end()].append((item, i, t))

                    # XXX The following 3 lines were commented out for causing a bug. See issue #768
                    # # Remove any items that successfully matched in this pass from the to_scan buffer.
                    # # This ensures we don't carry over tokens that already matched, if we're ignoring below.
                    # to_scan.remove(item)

            # 3) Process any ignores. This is typically used for e.g. whitespace.
            # We carry over any unmatched items from the to_scan buffer to be matched again after
            # the ignore. This should allow us to use ignored symbols in non-terminals to implement
            # e.g. mandatory spacing.
            for x in self.ignore:
                m = match(x, stream, i)
                if m:
                    # Carry over any items still in the scan buffer, to past the end of the ignored items.
                    delayed_matches[m.end()].extend(
                        [(item, i, None) for item in to_scan]
                    )

                    # If we're ignoring up to the end of the file, # carry over the start symbol if it already completed.
                    delayed_matches[m.end()].extend(
                        [
                            (item, i, None)
                            for item in columns[i]
                            if item.is_complete and item.s == start_symbol
                        ]
                    )

            next_to_scan = set()
            next_set = set()
            columns.append(next_set)
            transitives.append({})

            ## 4) Process Tokens from delayed_matches.
            # This is the core of the Earley scanner. Create an SPPF node for each Token,
            # and create the symbol node in the SPPF tree. Advance the item that completed,
            # and add the resulting new item to either the Earley set (for processing by the
            # completer/predictor) or the to_scan buffer for the next parse step.
            for item, _start, token in delayed_matches[i + 1]:
                if token is not None:
                    token.end_line = text_line
                    token.end_column = text_column + 1
                    token.end_pos = i + 1

                    new_item = item.advance()
                    label = (new_item.s, new_item.start, i)
                    token_node = TokenNode(token, terminals[token.type])
                    new_item.node = (
                        node_cache[label]
                        if label in node_cache
                        else node_cache.setdefault(label, SymbolNode(*label))
                    )
                    new_item.node.add_family(
                        new_item.s, item.rule, new_item.start, item.node, token_node
                    )
                else:
                    new_item = item

                if new_item.expect in self.TERMINALS:
                    # add (B ::= Aai+1.B, h, y) to Q'
                    next_to_scan.add(new_item)
                else:
                    # add (B ::= Aa+1.B, h, y) to Ei+1
                    next_set.add(new_item)

            del delayed_matches[i + 1]  # No longer needed, so unburden memory

            if not next_set and not delayed_matches and not next_to_scan:
                considered_rules = list(
                    sorted(to_scan, key=lambda key: key.rule.origin.name)
                )
                raise UnexpectedCharacters(
                    stream,
                    i,
                    text_line,
                    text_column,
                    {item.expect.name for item in to_scan},
                    set(to_scan),
                    state=frozenset(i.s for i in to_scan),
                    considered_rules=considered_rules,
                )

            return next_to_scan

        delayed_matches = defaultdict(list)
        match = self.term_matcher.match
        terminals = self.lexer_conf.terminals_by_name

        # Cache for nodes & tokens created in a particular parse step.
        transitives = [{}]

        text_line = 1
        text_column = 1

        ## The main Earley loop.
        # Run the Prediction/Completion cycle for any Items in the current Earley set.
        # Completions will be added to the SPPF tree, and predictions will be recursively
        # processed down to terminals/empty nodes to be added to the scanner for the next
        # step.
        i = 0
        for token in stream:
            self.predict_and_complete(i, to_scan, columns, transitives)

            to_scan = scan(i, to_scan)

            if token == "\n":
                text_line += 1
                text_column = 1
            else:
                text_column += 1
            i += 1

        self.predict_and_complete(i, to_scan, columns, transitives)

        ## Column is now the final column in the parse.
        assert i == len(columns) - 1
        return to_scan

    def predict_and_complete(self, i, to_scan, columns, transitives):
        """The core Earley Predictor and Completer.

        At each stage of the input, we handling any completed items (things
        that matched on the last cycle) and use those to predict what should
        come next in the input stream. The completions and any predicted
        non-terminals are recursively processed until we reach a set of,
        which can be added to the scan list for the next scanner cycle."""
        # Held Completions (H in E.Scotts paper).
        node_cache = {}
        held_completions = {}

        column = columns[i]
        # R (items) = Ei (column.items)
        items = deque(column)
        while items:
            item = items.pop()  # remove an element, A say, from R

            ### The Earley completer
            if item.is_complete:  ### (item.s == string)
                if item.node is None:
                    label = (item.s, item.start, i)
                    item.node = (
                        node_cache[label]
                        if label in node_cache
                        else node_cache.setdefault(label, SymbolNode(*label))
                    )
                    item.node.add_family(item.s, item.rule, item.start, None, None)

                # create_leo_transitives(item.rule.origin, item.start)

                ###R Joop Leo right recursion Completer
                if item.rule.origin in transitives[item.start]:
                    transitive = transitives[item.start][item.s]
                    if transitive.previous in transitives[transitive.column]:
                        root_transitive = transitives[transitive.column][
                            transitive.previous
                        ]
                    else:
                        root_transitive = transitive

                    new_item = Item(transitive.rule, transitive.ptr, transitive.start)
                    label = (root_transitive.s, root_transitive.start, i)
                    new_item.node = (
                        node_cache[label]
                        if label in node_cache
                        else node_cache.setdefault(label, SymbolNode(*label))
                    )
                    new_item.node.add_path(root_transitive, item.node)
                    if new_item.expect in self.TERMINALS:
                        # Add (B :: aC.B, h, y) to Q
                        to_scan.add(new_item)
                    elif new_item not in column:
                        # Add (B :: aC.B, h, y) to Ei and R
                        column.add(new_item)
                        items.append(new_item)
                ###R Regular Earley completer
                else:
                    # Empty has 0 length. If we complete an empty symbol in a particular
                    # parse step, we need to be able to use that same empty symbol to complete
                    # any predictions that result, that themselves require empty. Avoids
                    # infinite recursion on empty symbols.
                    # held_completions is 'H' in E.Scott's paper.
                    is_empty_item = item.start == i
                    if is_empty_item:
                        held_completions[item.rule.origin] = item.node

                    originators = [
                        originator
                        for originator in columns[item.start]
                        if originator.expect is not None and originator.expect == item.s
                    ]
                    for originator in originators:
                        new_item = originator.advance()
                        label = (new_item.s, originator.start, i)
                        new_item.node = (
                            node_cache[label]
                            if label in node_cache
                            else node_cache.setdefault(label, SymbolNode(*label))
                        )
                        new_item.node.add_family(
                            new_item.s, new_item.rule, i, originator.node, item.node
                        )
                        if new_item.expect in self.TERMINALS:
                            # Add (B :: aC.B, h, y) to Q
                            to_scan.add(new_item)
                        elif new_item not in column:
                            # Add (B :: aC.B, h, y) to Ei and R
                            column.add(new_item)
                            items.append(new_item)

            ### The Earley predictor
            elif item.expect in self.NON_TERMINALS:  ### (item.s == lr0)
                new_items = []
                for rule in self.predictions[item.expect]:
                    new_item = Item(rule, 0, i)
                    new_items.append(new_item)

                # Process any held completions (H).
                if item.expect in held_completions:
                    new_item = item.advance()
                    label = (new_item.s, item.start, i)
                    new_item.node = (
                        node_cache[label]
                        if label in node_cache
                        else node_cache.setdefault(label, SymbolNode(*label))
                    )
                    new_item.node.add_family(
                        new_item.s,
                        new_item.rule,
                        new_item.start,
                        item.node,
                        held_completions[item.expect],
                    )
                    new_items.append(new_item)

                for new_item in new_items:
                    if new_item.expect in self.TERMINALS:
                        to_scan.add(new_item)
                    elif new_item not in column:
                        column.add(new_item)
                        items.append(new_item)
