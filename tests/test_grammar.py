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
    "select count(`driver`) from w",
]

reject_queries = [
    # not_an_arg
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
    # Non-subquery in LLMQA arg
    """
    {{
        LLMQA(
            'What borough is the Kia Oval located in?',
            (
                this is not a query
            )
        )
    }}
    """,
    # Missing predicate arg
    """
    select * from w where x 'string';
    """,
]


@pytest.mark.parametrize("q", accept_queries)
def test_accept(parser, q):
    parser.parse(q)


@pytest.mark.parametrize("q", reject_queries)
def test_reject(parser, q):
    with pytest.raises(UnexpectedInput):
        parser.parse(q)
