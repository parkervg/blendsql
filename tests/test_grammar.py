import pytest
from blendsql.grammars.minEarley.parser import EarleyParser
from blendsql.grammars.minEarley.earley_exceptions import UnexpectedInput


@pytest.fixture(scope="session")
def parser() -> EarleyParser:
    return EarleyParser.open("./blendsql/grammars/_cfg_grammar.lark", start="start")


accept_queries = [
    """
    {{
        LLMQA(
            'What year was the player born?',
            (
                SELECT documents.title AS 'Player', documents.content FROM documents
                JOIN {{
                    LLMJoin(
                        left_on = 'w::winner',
                        right_on = 'documents::title'
                    )
                }}
                WHERE w.year = '1971-72'
            )
        )
    }}
    """,
    """
    {{
        LLMQA(
            'What borough is the Kia Oval located in?',
            (
                SELECT documents.title, documents.content FROM documents 
                JOIN {{
                    LLMJoin(
                        left_on='w::name',
                        right_on='documents::title'
                    )
                }} WHERE w.name = 'kia oval'
            )
        )
    }}
    """,
]

reject_queries = [
    """
    {{
        LLMQA(
            'What year was the player born?',
            (
                SELECT documents.title AS 'Player', documents.content FROM documents
                JOIN {{
                    LLMJoin(
                        not_an_arg = 'w::winner',
                        right_on = 'documents::title'
                    )
                }}
                WHERE w.year = '1971-72'
            )
        )
    }}
    """,
    # Missing parentheses
    """
    {{
        LLMQA(
            'What borough is the Kia Oval located in?'
            (
                SELECT documents.title, documents.content FROM documents 
                JOIN {{
                    LLMJoin(
                        left_on='w::name',
                        right_on='documents::title'
                    )
                }} WHERE w.name = 'kia oval'
            )
        )
    }}
    """,
]


def test_accept(parser):
    for q in accept_queries:
        parser.parse(q)


def test_reject(parser):
    for q in reject_queries:
        with pytest.raises(UnexpectedInput):
            parser.parse(q)
