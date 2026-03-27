import datetime
import pytest
import pandas as pd

import blendsql
from blendsql.pandas_api import _resolve_return_type
from blendsql.types.types import DataTypes


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_name": ["John Wall", "Jayson Tatum"],
            "Report": [
                "He had 2 assists and 26 points",
                "He only had 1 assist but scored 51 points",
            ],
            "AnotherOptionalReport": ["He had 26pts", "He scored 51pts!"],
        }
    )


def test_resolve_none():
    assert _resolve_return_type(None) is None


def test_resolve_str_passthrough():
    assert _resolve_return_type("int") == "int"
    assert _resolve_return_type("float") == "float"
    assert _resolve_return_type("str") == "str"
    assert _resolve_return_type("bool") == "bool"
    assert _resolve_return_type("date") == "date"
    assert _resolve_return_type("substring") == "substring"


def test_resolve_python_scalar_types():
    assert _resolve_return_type(int) == "int"
    assert _resolve_return_type(float) == "float"
    assert _resolve_return_type(str) == "str"
    assert _resolve_return_type(bool) == "bool"
    assert _resolve_return_type(datetime.date) == "date"


def test_resolve_python_list_types():
    assert _resolve_return_type(list[int]) == "list[int]"
    assert _resolve_return_type(list[float]) == "list[float]"
    assert _resolve_return_type(list[str]) == "list[str]"
    assert _resolve_return_type(list[bool]) == "list[bool]"
    assert _resolve_return_type(list[datetime.date]) == "list[date]"


def test_resolve_datatype_passthrough():
    dt = DataTypes.INT()
    assert _resolve_return_type(dt) is dt
    dt_list = DataTypes.STR("*")
    assert _resolve_return_type(dt_list) is dt_list


def test_resolve_unknown_type_falls_back_to_str():
    class _Custom:
        pass

    assert _resolve_return_type(_Custom) == "str"


def test_llmmap_scalar_int(df, model):
    blendsql.config(model=model)
    result = df.llmmap(
        "How many points did the player score?",
        context_cols=["player_name", "Report"],
        return_type=int,
    )
    assert len(result) == len(df)
    assert all(isinstance(v, (int, float, type(None))) for v in result)


def test_llmmap_scalar_str(df, model):
    blendsql.config(model=model)
    result = df.llmmap(
        "What is the player's name?",
        context_cols=["player_name", "Report"],
        return_type=str,
    )
    assert len(result) == len(df)
    assert all(isinstance(v, (str, type(None))) for v in result)


def test_llmmap_list_int(df, model):
    blendsql.config(model=model)
    result = df.llmmap(
        "List the player's points and assists in the order [points, assists].",
        context_cols=["player_name", "Report"],
        return_type=list[int],
    )
    assert len(result) == len(df)
    for row in result:
        if row is not None:
            assert isinstance(row, list)
            assert all(isinstance(v, (int, float, type(None))) for v in row)


def test_llmmap_single_context_col(df, model):
    blendsql.config(model=model)
    result = df.llmmap(
        "How many assists did the player have?",
        context_cols=["Report"],
        return_type=int,
    )
    assert len(result) == len(df)


def test_llmmap_options(df, model):
    blendsql.config(model=model)
    result = df.llmmap(
        "Did the player score more than 20 points?",
        context_cols=["player_name", "Report"],
        options=["yes", "no"],
    )
    assert len(result) == len(df)
    assert all(v in ("yes", "no", None) for v in result)


def test_llmmap_invalid_column_raises(df, model):
    blendsql.config(model=model)
    with pytest.raises(ValueError, match="not in dataframe"):
        df.llmmap(
            "How many points?",
            context_cols=["nonexistent_column"],
        )


def test_llmmap_returns_series_assignable(df, model):
    blendsql.config(model=model)
    result = df.llmmap(
        "How many points did the player score?",
        context_cols=["player_name", "Report"],
        return_type=int,
    )
    out = df.copy()
    out["points"] = result
    assert "points" in out.columns
    assert len(out) == len(df)
