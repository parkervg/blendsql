import pytest

from blendsql import blend, ImageCaption
from blendsql._smoothie import Smoothie
from blendsql.db import SQLite
from blendsql.utils import fetch_from_hub
from blendsql.models import TransformersVisionModel

TEST_TRANSFORMERS_VISION_LLM = "Mozilla/distilvit"


@pytest.fixture(scope="session")
def db() -> SQLite:
    return SQLite(fetch_from_hub("national_parks.db"))


@pytest.fixture(scope="session")
def ingredients() -> set:
    return {ImageCaption}


@pytest.fixture(scope="session")
def model() -> TransformersVisionModel:
    return TransformersVisionModel(TEST_TRANSFORMERS_VISION_LLM, caching=False)


@pytest.mark.long
def test_no_ingredients(db, model, ingredients):
    res = blend(
        query="""
        select * from parks
        """,
        db=db,
        default_model=model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)


@pytest.mark.long
def test_image_caption(db, model, ingredients):
    res = blend(
        query="""
        SELECT "Name",
        {{ImageCaption('parks::Image')}} as "Image Description"
        FROM parks
        WHERE "Location" = 'Alaska'
        """,
        db=db,
        default_model=model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)
