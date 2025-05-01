import pytest
import pandas as pd
from blendsql import BlendSQL
from blendsql.db import Pandas
from blendsql.common.exceptions import IngredientException, InvalidBlendSQL
from blendsql.ingredients import LLMQA
from tests.utils import select_first_option


@pytest.fixture(scope="session")
def bsql() -> BlendSQL:
    return BlendSQL(
        db=Pandas(
            {
                "w": pd.DataFrame(
                    {"Name": ["Danny", "Emma", "Tony"], "Age": [23, 26, 19]}
                ),
                "classes": pd.DataFrame(
                    {
                        "Id": [4532, 1234, 6653],
                        "Title": ["Computer Science", "Discrete Math", "Philosophy"],
                    }
                ),
            }
        ),
    )


def test_error_on_delete1(bsql):
    with pytest.raises(InvalidBlendSQL):
        _ = bsql.execute(
            query="""
            DELETE FROM w WHERE TRUE;
            """,
        )


def test_error_on_delete2(bsql):
    with pytest.raises(InvalidBlendSQL):
        _ = bsql.execute(
            query="""
            DROP TABLE w;
            """,
        )


def test_error_on_invalid_ingredient(bsql):
    with pytest.raises(IngredientException):
        _ = bsql.execute(
            query="""
            SELECT * w WHERE {{ingredient()}} = TRUE
            """,
            ingredients={"This is not an ingredient type"},
        )


def test_error_on_bad_options_subquery(bsql):
    with pytest.raises(InvalidBlendSQL):
        _ = bsql.execute(
            query="""
            SELECT * FROM w
            WHERE {{
                select_first_option(
                    'I am at a nice cafe right now',
                    'w::Name',
                    options=(SELECT * FROM w)
                )
            }}
            """,
            ingredients={select_first_option},
        )


def test_replacement_scan(bsql, constrained_model):
    """ad94437"""
    NewIngredient = LLMQA.from_args(k=2)
    _ = bsql.execute(
        """
        SELECT * FROM w 
        WHERE w.Name = {{NewIngredient('Will this work?')}}
        """,
        ingredients={NewIngredient},
        model=constrained_model,
    )


def test_llmqa_question_f_strings(bsql, constrained_model):
    """0218f7f"""
    res = bsql.execute(
        """
        WITH t AS (SELECT * FROM w WHERE Age = 23)
        SELECT {{
            LLMQA(
                'Please say "{t::Name}"'
            )    
        }}
        """,
        model=constrained_model,
    )
    assert list(res.df.values.flat) == ["Danny"]


def test_llmmap_question_f_strings(bsql, constrained_model):
    """0218f7f"""
    _ = bsql.execute(
        """
        WITH t AS (SELECT * FROM w WHERE Age = 23)
        SELECT c.Title, {{LLMMap('Would someone named {t::Name} be good at this subject?', 'c::Title')}} AS "answer" FROM classes c
        """
    )
