import pytest
import pandas as pd

from blendsql.common.utils import fetch_from_hub
from blendsql import BlendSQL


@pytest.fixture(scope="module")
def bsql() -> BlendSQL:
    return BlendSQL(
        {
            "w": pd.DataFrame(
                {
                    "player_name": ["John Wall", "Jayson Tatum"],
                    "Report": ["He had 2 assists", "He only had 1 assist"],
                    "AnotherOptionalReport": ["He had 26pts", "He scored 51pts!"],
                }
            ),
            "v": pd.DataFrame({"people": ["john", "jayson", "emily"]}),
            "names_and_ages": pd.DataFrame(
                {
                    "Name": ["Tommy", "Sarah", "Tommy"],
                    "Description": ["He is 24 years old", "She's 12", "He's only 3"],
                }
            ),
            "movie_reviews": pd.DataFrame(
                {"review": ["I love this movie!", "This was SO GOOD"]}
            ),
        },
    )


def test_map_to_columns(bsql, model):
    smoothie = bsql.execute(
        """
        WITH player_stats AS (
            SELECT *, {{
            LLMMap(
                'How many points and assists did {} have? Respond in the order [points, assists].', 
                player_name, 
                Report, 
                return_type='List[int]',
                quantifier='{2}'
                )
            }} AS box_score_values
            FROM w
        ) SELECT 
        player_name,
        list_element(box_score_values, 1) AS points,
        list_element(box_score_values, 2) AS assists
        FROM player_stats
        """,
        model=model,
    )
    assert smoothie.df().columns.tolist() == ["player_name", "points", "assists"]


def test_map_context_with_duplicate_values(bsql, model):
    smoothie = bsql.execute(
        """
         SELECT *, {{
            LLMMap(
                'How old is {}?',
                Name,
                Description,
                return_type='int'
            )
        }} FROM "names_and_ages"
        """,
        model=model,
    )
    df = smoothie.df()
    assert set(df[df["Name"] == "Tommy"]["How old is {}?"].values.tolist()) == {24, 3}
    assert df[df["Description"] == "He's only 3"]["How old is {}?"].values.item() == 3


def test_map_options_to_string(bsql, model):
    """cc00959"""
    smoothie = bsql.execute(
        """
         SELECT * FROM movie_reviews 
         WHERE {{
            LLMMap(
                'What is the sentiment of this review?',
                review,
                options=('POSITIVE', 'NEGATIVE')
            )
        }} = 'POSITIVE'
        """,
        model=model,
    )
    assert not smoothie.df().empty


def test_complex_nested_cte(model):
    """43c71bb"""
    bsql = BlendSQL(
        fetch_from_hub("ecomm/sf_500/ecomm_database_500.duckdb"),
        model=model,
    )
    _ = bsql.execute(
        """
        WITH images AS (
            SELECT
                sd.id,
                sd.productDisplayName AS title,
                sd.productDescriptors.description.value AS descr,
                sd.price,
                img.link AS image_path
            FROM styles_details sd
            JOIN image_mapping img ON CAST(sd.id AS VARCHAR) = img.id
            WHERE {{
                LLMMap('Is black the dominant color on this product?', image_path)
            }} AND baseColour = 'Black' LIMIT 5
        ),
        img_classifications AS (
            SELECT *,
            {{
                LLMMap(
                    'Classify the clothing item in the image. If there are multiple products in the picture, always refer to the most prominent one.
                    ''shoes'' and things like sandals, flip-flops, or other shoes.
                    ''bottoms'' are pieces of apparel that can be worn on the lower part of the body, like pants, shorts, and skirts, but NOT swimwear.
                    ''swimwear'' are things meant to be worn while swimming.
                    ''tops'' are pieces of apparel that can be worn on the upper part of the body, like t-shirts, shirts, pullovers, hoodies, but still requires some sort of clothing on the lower body (i.e., not a dress).
                    ''accessories'' are things like jewelry or a bag, including handbags or a (gym) backpacks',
                    image_path,
                    options=('shoes', 'bottoms', 'swimwear', 'tops', 'accessories', 'N.A.')
                )
            }} AS category,
            {{
                LLMMap(
                    'What is the brand name of this product? Return just the brand name.',
                    title, descr
                )
            }} AS brand
            FROM images
            WHERE category IN ('shoes', 'bottoms', 'tops', 'accessories')
        ),
        pairings AS (
            SELECT
                s.id AS id1, s.title AS title1, s.image_path AS image1,
                b.id AS id2, b.title AS title2, b.image_path AS image2,
                t.id AS id3, t.title AS title3, t.image_path AS image3,
                a.id AS id4, a.title AS title4, a.image_path AS image4
            FROM img_classifications s
            JOIN img_classifications b ON LOWER(s.brand) = LOWER(b.brand)
            JOIN img_classifications t ON LOWER(s.brand) = LOWER(t.brand)
            JOIN img_classifications a ON LOWER(s.brand) = LOWER(a.brand)
            WHERE s.category = 'shoes'
              AND b.category = 'bottoms'
              AND t.category = 'tops'
              AND a.category = 'accessories'
              AND a.price <= 500
        )
        SELECT id1 || '-' || id2 || '-' || id3 || '-' || id4 AS id
        FROM pairings
        """
    )


def test_map_with_unexpected_options(bsql, model):
    """A test to see if our option constraints are being respected"""
    smoothie = bsql.execute(
        """
         SELECT *, {{
            LLMMap(
                'How old is {}?',
                Name,
                Description,
                options=("Do NOT say this!")
            )
        }} AS response FROM "names_and_ages"
        """,
        model=model,
    )
    df = smoothie.df()
    assert all([x == "Do NOT say this!" for x in df["response"]])
