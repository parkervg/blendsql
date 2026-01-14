import pytest
import pandas as pd

from blendsql import BlendSQL, config
from blendsql.ingredients import LLMMap
from blendsql.models import ConstrainedModel

config.set_async_limit(1)


@pytest.fixture(scope="module")
def bsql() -> BlendSQL:
    return BlendSQL(
        {
            "People": pd.DataFrame(
                {
                    "Name": [
                        "George Washington",
                        "John Quincy Adams",
                        "Thomas Jefferson",
                        "James Madison",
                        "James Monroe",
                        "Alexander Hamilton",
                        "Sabrina Carpenter",
                        "Charli XCX",
                        "Elon Musk",
                        "Michelle Obama",
                    ],
                    "Known_For": [
                        "Established federal government, First U.S. President",
                        "XYZ Affair, Alien and Sedition Acts",
                        "Louisiana Purchase, Declaration of Independence",
                        "War of 1812, Constitution",
                        "Monroe Doctrine, Missouri Compromise",
                        "Created national bank, Federalist Papers",
                        "Nonsense, Emails I Cant Send, Mean Girls musical",
                        "Crash, How Im Feeling Now, Boom Clap",
                        "Tesla, SpaceX, Twitter/X acquisition",
                        "Lets Move campaign, Becoming memoir",
                    ],
                }
            ),
            "Eras": pd.DataFrame({"Years": ["1800-1900", "1900-2000", "2000-Now"]}),
        },
    )


def test_num_values_passed_map(bsql, model):
    # With the above db, this will be 10
    total_num_values_to_process = bsql.db.execute_to_list(
        "SELECT COUNT(DISTINCT Name) FROM People", to_type=int
    )[0]
    max_tokens = 1  # since 'y' and 'n' are one token each
    for bs in [1, 5, 10]:
        assert model.completion_tokens == 0
        assert model.prompt_tokens == 0
        print(f"{bs=}")
        smoothie = bsql.execute(
            f"""
            SELECT {{{{LLMMap('Is a famous singer?', p.Name, options=('y', 'n'))}}}} FROM People p
            """,
            ingredients={LLMMap.from_args(batch_size=bs)},
            model=model,
        )
        assert smoothie.meta.num_values_passed == total_num_values_to_process
        assert smoothie.meta.num_generation_calls == total_num_values_to_process
        if isinstance(
            model, ConstrainedModel
        ):  # Unconstrained models will add 'SEP', messing up exact token counts
            assert smoothie.meta.completion_tokens == (
                max_tokens * total_num_values_to_process
            ), f"{smoothie.meta.completion_tokens=}, {(max_tokens * total_num_values_to_process)=}, {bs=}"


def test_num_values_passed_qa(bsql, model):
    # With the above db, this will be 10
    smoothie = bsql.execute(
        """
        SELECT * FROM People p 
        WHERE p.Name = {{LLMQA('Who founded Tesla?')}}
        OR p.Known_For = {{LLMQA('Which of these do you like best?')}}
        """,
        model=model,
    )
    assert smoothie.meta.num_values_passed == 0
    assert smoothie.meta.num_generation_calls == 2
