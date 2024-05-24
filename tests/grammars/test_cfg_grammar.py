import pytest
from blendsql import LLMQA, LLMJoin, LLMMap
from blendsql.grammars.utils import load_cfg_parser, EarleyParser
from blendsql.grammars.minEarley.earley_exceptions import UnexpectedInput


@pytest.fixture(scope="session")
def parser() -> EarleyParser:
    return load_cfg_parser({LLMQA, LLMJoin, LLMMap})


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
    """
    {{
        LLMQA(
            'In what event was the cyclist ranked highest?',
            (
                SELECT * FROM (
                    SELECT * FROM "./Cuba at the UCI Track Cycling World Championships (2)"
                ) as w WHERE w.name = {{
                    LLMQA(
                        "Which cyclist was born in Matanzas, Cuba?",
                        (
                            SELECT * FROM documents 
                                WHERE documents MATCH 'matanzas AND (cycling OR track OR born)' 
                                ORDER BY rank LIMIT 3
                        ),
                        options="w::name"
                    )
                }}
            ),
            options='w::event'
        )
    }}
    """,
    """
    {{
        LLMQA(
            'When was the Rangers Player born?',
            (
                WITH t AS (
                    SELECT player FROM (
                        SELECT * FROM "./List of Rangers F.C. records and statistics (0)"
                        UNION ALL SELECT * FROM "./List of Rangers F.C. records and statistics (1)"
                    ) ORDER BY trim(fee, '£') DESC LIMIT 1 OFFSET 2
                ), d AS (
                    SELECT * FROM documents JOIN t WHERE documents MATCH t.player || ' OR rangers OR fc' ORDER BY rank LIMIT 5
                ) SELECT d.content, t.player AS 'Rangers Player' FROM d JOIN t
            )
        )
    }}
    """,
    # Named arguments should work
    """
    {{
        LLMQA(
            question='Tell me why the sky is blue',
            context=(SELECT * FROM w LIMIT 10)
        )
    }}
    """,
    """
    SELECT * FROM w WHERE 1+1 = 2 AND {{LLMMap('more than 30 points?', 'w::score')}} = TRUE
    """,
    "select count(`driver`) from w",
    "with t as (SELECT count(`driver`) from w), d as (SELECT * FROM w) SELECT * FROM t;",
    "select w.driver || w.constructor from w",
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
    # Duplicate 'with' in CTE
    "with t as (SELECT count(`driver`) from w), with d as (SELECT * FROM w) SELECT * FROM t;",
    # Invalid use of concat
    "select driver from w || t;",
    # Can't use LLMMap here
    "SELECT * FROM w JOIN {{LLMMap('more than 30 points?', 'table::column')}}",
]


@pytest.mark.parametrize("q", accept_queries)
def test_accept(parser, q):
    parser.parse(q)


@pytest.mark.parametrize("q", reject_queries)
def test_reject(parser, q):
    with pytest.raises(UnexpectedInput):
        parser.parse(q)


def test_unspecified_ingredient_reject():
    """We pass in LLMQA to load_cfg_parser, but then use LLMap.
    This should raise a grammar error.
    """
    parser = load_cfg_parser(ingredients={LLMQA})
    with pytest.raises(UnexpectedInput):
        parser.parse(
            """
            SELECT * FROM w WHERE {{LLMMap('more than 30 points?', 'w::score')}} = TRUE
            """
        )


def test_no_ingredients_reject():
    """If we don't pass any ingredients into load_cfg_parser,
    we essentially just have a SQLite CFG grammar.
    """
    parser = load_cfg_parser()
    with pytest.raises(UnexpectedInput):
        parser.parse(
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
            """
        )
