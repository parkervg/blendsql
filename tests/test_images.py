import pytest
import pandas as pd

from blendsql import BlendSQL


@pytest.fixture(scope="module")
def bsql() -> BlendSQL:
    return BlendSQL(
        {
            "Colors": pd.DataFrame(
                [
                    {
                        "name": "Red",
                        "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOEAAADhCAMAAAAJbSJIAAAAA1BMVEX/AAAZ4gk3AAAASElEQVR4nO3BgQAAAADDoPlTX+AIVQEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADwDcaiAAFXD1ujAAAAAElFTkSuQmCC",
                    },
                    {
                        "name": "Blue",
                        "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAQMAAADCCAMAAAB6zFdcAAAAA1BMVEUAf/8i37duAAAASElEQVR4nO3BMQEAAADCoPVPbQwfoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIC3AcUIAAFkqh/QAAAAAElFTkSuQmCC",
                    },
                ]
            ),
        },
    )


def test_simple_image(bsql, model):
    smoothie = bsql.execute(
        """
        SELECT name, {{LLMMap('What color is this?', image)}} AS prediction
        FROM Colors
        """,
        model=model,
    )
    assert [i.lower() for i in smoothie.df()["name"].tolist()] == [
        i.lower() for i in smoothie.df()["prediction"].tolist()
    ]
