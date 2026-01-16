import pytest
import pandas as pd

from blendsql import BlendSQL, config
from blendsql.models import ConstrainedModel

config.set_async_limit(1)
config.set_deterministic(True)


@pytest.fixture(scope="module")
def bsql() -> BlendSQL:
    return BlendSQL(
        {
            "Reviews": pd.DataFrame(
                {
                    "reviewText": [
                        "You'll like this one - it has single quotes",
                        "This'll also float your boat",
                        "Don't wait on this film",
                    ],
                    "hasSingleQuote": [True, True, True],
                    "Year": [2025, 2025, 2025],
                }
            )
        }
    )


def test_qa_tuple_with_single_quotes(bsql, model):
    if not isinstance(model, ConstrainedModel):
        pytest.skip()
    smoothie = bsql.execute(
        """
            SELECT * FROM (
                VALUES {{
                    LLMQA(
                        'Select the 3 most negative reviews.',
                        options=(SELECT reviewText FROM Reviews WHERE Year = 2025),
                        return_type='List[str]',
                        quantifier='{3}'
                    )
                }}
            ) AS rankedReviews(review1, review2, review3)
        """,
        model=model,
    )
    assert smoothie.df.columns.tolist() == ["review1", "review2", "review3"]
