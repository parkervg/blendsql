import pandas as pd

from blendsql import BlendSQL
from blendsql.ingredients import LLMJoin, LLMQA
from blendsql.models import LiteLLM

if __name__ == "__main__":
    bsql = BlendSQL(
        {
            "fruit": pd.DataFrame(
                {"name": ["apple", "orange", "apricot", "pear", "blueberry", "banana"]}
            ),
            "colors": pd.DataFrame(
                {"name": ["orange", "blue", "yellow", "red", "yellow"]}
            ),
        },
        ingredients={LLMJoin, LLMQA},
        model=LiteLLM("openai/gpt-4o-mini", caching=False),
        verbose=True,
    )
    smoothie = bsql.execute(
        """
        SELECT * FROM fruit
        JOIN {{
            LLMJoin(
                'fruit::name',
                'colors::name'
            )
        }}
        """,
    )
