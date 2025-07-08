from lark import Lark
from lark.parsers.lalr_interactive_parser import InteractiveParser
from lark.lexer import PatternRE, PatternStr, Token
from dataclasses import dataclass
from typing import List
import heapq

from blendsql.ingredients import LLMQA, LLMMap
from load_grammar import load_grammar

if __name__ == "__main__":
    LLMSearchMap = LLMMap.from_args()
    parser = Lark(
        load_grammar(
            open("./grammar.lark").read(),
            ingredients=[LLMQA, LLMMap, LLMSearchMap],
        ),
        parser="lalr",
    )
    terminal_map = {r.name: r for r in parser.terminals}

    @dataclass
    class GrammarBeam:
        score: int
        tokens: List[Token]
        curr_pointer: int
        parser: InteractiveParser

        def __lt__(self, other):
            return self.score < other.score

    # Initialize text and parameters
    text = "SELECT {{LLMQA('a question'))}}"
    max_width = 5
    max_score = 5

    # Initialize beams and results
    # first, see if we can even lex our input
    lexed_tokens = None
    while lexed_tokens is None:
        try:
            lexed_tokens = list(parser.lex(text))
        except:
            text = text[:-1]
    initial_beam = GrammarBeam(0, lexed_tokens, 0, parser.parse_interactive())
    active_beams = [initial_beam]
    results = []

    # Perform beam search iteratively
    while active_beams:
        # Get beam with lowest score
        beam = heapq.heappop(active_beams)

        # Skip if score exceeds maximum
        if beam.score > max_score:
            continue

        curr_pointer = beam.curr_pointer
        p = beam.parser

        # Check if we've reached the end of tokens
        if curr_pointer >= len(beam.tokens):
            try:
                p.feed_eof()
                results.append(beam)
            except:
                pass
            continue

        t = beam.tokens[curr_pointer]

        # Check for special function tokens first
        new_beams_from_functions = []
        for function_name in ["LLMMAP", "LLMQA", "LLMJOIN"]:
            if function_name in p.accepts():
                new_tokens = beam.tokens.copy()
                new_tokens[curr_pointer] = Token(
                    function_name, terminal_map[function_name].pattern.value
                )
                new_parser = p.copy()
                try:
                    new_parser.feed_token(new_tokens[curr_pointer])
                    new_beam = GrammarBeam(
                        beam.score + 1, new_tokens, curr_pointer + 1, new_parser
                    )
                    new_beams_from_functions.append(new_beam)
                except:
                    pass

        # If we have function-specific beams, prioritize them
        if new_beams_from_functions:
            for new_beam in new_beams_from_functions:
                heapq.heappush(active_beams, new_beam)
            continue

        # Try to accept the current token
        if t.type in p.accepts():
            # Continue current beam
            new_parser = p.copy()
            try:
                new_parser.feed_token(t)
                new_beam = GrammarBeam(
                    beam.score, beam.tokens, curr_pointer + 1, new_parser
                )
                heapq.heappush(active_beams, new_beam)
                continue
            except:
                pass

        # Create new beams based on error correction
        new_beams = []

        for accept_t in p.accepts():
            if accept_t == "$END":  # End of parsing
                try:
                    end_parser = p.copy()
                    end_parser.feed_eof()
                    results.append(beam)
                except:
                    pass
                continue

            pattern = terminal_map[accept_t].pattern
            if isinstance(pattern, PatternStr):
                insert_token = Token(accept_t, terminal_map[accept_t].pattern.value)

                # Replace operation
                new_tokens = beam.tokens.copy()
                new_tokens[curr_pointer] = insert_token
                new_parser = p.copy()
                try:
                    new_parser.feed_token(insert_token)
                    new_beams.append(
                        GrammarBeam(
                            beam.score + 1, new_tokens, curr_pointer + 1, new_parser
                        )
                    )
                except:
                    pass

                # Insert operation
                new_tokens = beam.tokens.copy()
                new_tokens = (
                    new_tokens[:curr_pointer]
                    + [insert_token]
                    + new_tokens[curr_pointer:]
                )
                new_parser = p.copy()
                try:
                    new_parser.feed_token(insert_token)
                    new_beams.append(
                        GrammarBeam(
                            beam.score + 1, new_tokens, curr_pointer + 1, new_parser
                        )
                    )
                except:
                    pass

                # Delete operation
                if curr_pointer < len(beam.tokens) - 1:
                    new_tokens = beam.tokens.copy()
                    new_tokens = (
                        new_tokens[:curr_pointer] + new_tokens[curr_pointer + 1 :]
                    )
                    new_beams.append(
                        GrammarBeam(beam.score + 1, new_tokens, curr_pointer, p.copy())
                    )

            elif not isinstance(pattern, PatternRE):
                raise ValueError(f"Unexpected pattern type: {type(pattern)}")

        # Prune beams and add to active beams
        new_beams.sort(key=lambda x: x.score)
        for new_beam in new_beams[:max_width]:
            heapq.heappush(active_beams, new_beam)

    # Return the best result
    if results:
        sorted_results = sorted(results, key=lambda r: r.score)
        for r in sorted_results[:3]:
            print(" ".join([i.value for i in r.tokens]))
    else:
        print("No valid parse found")
