import pytest
import os

from blendsql import BlendSQL
from blendsql.ingredients import ImageCaption, LLMMap, LLMQA, RAGQA
from blendsql._smoothie import Smoothie
from blendsql.utils import fetch_from_hub
from blendsql.models import LiteLLM
from blendsql._exceptions import IngredientException


@pytest.fixture(scope="session")
def bsql(vision_model) -> BlendSQL:
    return BlendSQL(fetch_from_hub("national_parks.db"))


@pytest.mark.long
def test_no_ingredients(bsql):
    res = bsql.execute(
        """
        select * from parks
        """,
    )
    assert isinstance(res, Smoothie)


@pytest.mark.long
def test_image_caption(bsql, vision_model):
    ingredients = {ImageCaption}
    res = bsql.execute(
        """
        SELECT "Name",
        {{ImageCaption('parks::Image')}} as "Image Description"
        FROM parks
        WHERE "Location" = 'Alaska'
        """,
        ingredients=ingredients,
        model=vision_model,
    )
    assert isinstance(res, Smoothie)


@pytest.mark.long
def test_mixed_models(bsql, vision_model, model):
    ingredients = {ImageCaption.from_args(model=vision_model), LLMMap}
    res = bsql.execute(
        """
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
        model=model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)


@pytest.mark.long
def test_mixed_models_no_default_error(bsql, vision_model):
    """If we don't pass a model and don't specify one in `.from_args`,
    we should raise the appropriate IngredientException error.
    """
    ingredients = {ImageCaption.from_args(model=vision_model), LLMMap}
    with pytest.raises(IngredientException):
        _ = bsql.execute(
            """
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
            ingredients=ingredients,
        )


@pytest.mark.long
def test_readme_example_1(bsql, model):
    ingredients = {LLMMap}
    res = bsql.execute(
        """
            SELECT "Name", "Description" FROM parks
            WHERE {{
                LLMMap(
                    'Does this location have park facilities?',
                    context='parks::Description'
                )
            }} = FALSE
            """,
        model=model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)


@pytest.mark.long
def test_readme_example_2(bsql, model):
    ingredients = {LLMQA}
    res = bsql.execute(
        """
            SELECT "Location", "Name" AS "Park Protecting Ash Flow" FROM parks
            WHERE "Name" = {{
              LLMQA(
                'Which park protects an ash flow?',
                context=(SELECT "Name", "Description" FROM parks),
                options="parks::Name"
              )
          }}
            """,
        model=model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)


@pytest.mark.long
def test_readme_example_3(bsql, model):
    ingredients = {LLMMap}
    res = bsql.execute(
        """
        SELECT COUNT(*) FROM parks
            WHERE {{LLMMap('How many states?', 'parks::Location')}} > 1
            """,
        model=model,
        ingredients=ingredients,
    )
    assert isinstance(res, Smoothie)


@pytest.mark.long
def test_rag_qa(bsql, model):
    if os.getenv("AZURE_SUBSCRIPTION_KEY") is None:
        pytest.skip("AZURE_SUBSCRIPTION_KEY environment variable not set")
    res = bsql.execute(
        """
        SELECT Name, Location, Description FROM parks
        WHERE Location = {{RAGQA('Which state was Sarah Palin governor of?')}}
        """,
        model=model,
        # We expect the `blend()` call to fetch the appropriate ingredients for us
        ingredients={RAGQA},
    )
    assert isinstance(res, Smoothie)
    if isinstance(model, (LiteLLM)):
        assert res.df["Location"].unique().tolist() == ["Alaska"]
