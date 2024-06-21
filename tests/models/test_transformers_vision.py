import pytest

from blendsql import blend, ImageCaption, LLMMap, LLMQA
from blendsql._smoothie import Smoothie
from blendsql.db import SQLite
from blendsql.utils import fetch_from_hub
from blendsql.models import TransformersVisionModel, TransformersLLM
from blendsql._exceptions import IngredientException

TEST_TRANSFORMERS_LLM = "hf-internal-testing/tiny-random-PhiForCausalLM"
TEST_TRANSFORMERS_VISION_LLM = "Mozilla/distilvit"


@pytest.fixture(scope="session")
def db() -> SQLite:
    return SQLite(fetch_from_hub("national_parks.db"))


@pytest.fixture(scope="session")
def vision_model() -> TransformersVisionModel:
    return TransformersVisionModel(TEST_TRANSFORMERS_VISION_LLM, caching=False)


@pytest.fixture(scope="session")
def text_model() -> TransformersLLM:
    return TransformersLLM(TEST_TRANSFORMERS_LLM, caching=False)


@pytest.mark.long
def test_no_ingredients(db, vision_model):
    res = blend(
        query="""
        select * from parks
        """,
        db=db,
        default_model=vision_model,
    )
    assert isinstance(res, Smoothie)


@pytest.mark.long
def test_image_caption(db, vision_model):
    ingredients = {ImageCaption}
    res = blend(
        query="""
        SELECT "Name",
        {{ImageCaption('parks::Image')}} as "Image Description"
        FROM parks
        WHERE "Location" = 'Alaska'
        """,
        db=db,
        default_model=vision_model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)


@pytest.mark.long
def test_mixed_models(db, vision_model, text_model):
    ingredients = {ImageCaption.from_args(model=vision_model), LLMMap}
    res = blend(
        query="""
        SELECT "Name",
        {{ImageCaption('parks::Image')}} as "Image Description", 
        {{
            LLMMap(
                question='Size in km2?',
                context='parks::Area'
            )
        }} as "Size in km" FROM parks
        WHERE "Location" = 'Alaska'
        ORDER BY "Size in km" DESC LIMIT 1
            """,
        db=db,
        default_model=text_model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)


@pytest.mark.long
def test_mixed_models_no_default_error(db, vision_model):
    """If we don't pass a default_model and don't specify one in `.from_args`,
    we should raise the appropriate IngredientException error.
    """
    ingredients = {ImageCaption.from_args(model=vision_model), LLMMap}
    with pytest.raises(IngredientException):
        _ = blend(
            query="""
            SELECT "Name",
            {{ImageCaption('parks::Image')}} as "Image Description", 
            {{
                LLMMap(
                    question='Size in km2?',
                    context='parks::Area'
                )
            }} as "Size in km" FROM parks
            WHERE "Location" = 'Alaska'
            ORDER BY "Size in km" DESC LIMIT 1
                """,
            db=db,
            ingredients=ingredients,
        )


@pytest.mark.long
def test_readme_example_1(db, text_model):
    ingredients = {LLMMap}
    res = blend(
        query="""
            SELECT "Name", "Description" FROM parks
            WHERE {{
                LLMMap(
                    'Does this location have park facilities?',
                    context='parks::Description'
                )
            }} = FALSE
            """,
        db=db,
        default_model=text_model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)


@pytest.mark.long
def test_readme_example_2(db, text_model):
    ingredients = {LLMQA}
    res = blend(
        query="""
            SELECT "Location", "Name" AS "Park Protecting Ash Flow" FROM parks 
            WHERE "Name" = {{
              LLMQA(
                'Which park protects an ash flow?',
                context=(SELECT "Name", "Description" FROM parks),
                options="parks::Name"
              ) 
          }}
            """,
        db=db,
        default_model=text_model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)


@pytest.mark.long
def test_readme_example_3(db, text_model):
    ingredients = {LLMMap}
    res = blend(
        query="""
        SELECT COUNT(*) FROM parks
            WHERE {{LLMMap('How many states?', 'parks::Location')}} > 1
            """,
        db=db,
        default_model=text_model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)
