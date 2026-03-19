import pandas as pd

from blendsql import BlendSQL
from blendsql.models import VLLM

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
        model=VLLM(
            model_name_or_path="RedHatAI/gemma-3-12b-it-quantized.w4a16",
            base_url="http://127.0.0.1:8000/v1/",
        ),
        verbose=True,
    )
    smoothie = bsql.execute(
        """
        SELECT f.name, c.name FROM fruit f
        JOIN colors c ON {{
            LLMJoin(
                f.name,
                c.name,
                join_criteria='Join the fruit to its color.'
            )
        }}
        """,
    )
    smoothie.print_summary()
