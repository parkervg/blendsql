import pandas as pd

from blendsql import BlendSQL
from blendsql.models import VLLM

if __name__ == "__main__":
    bsql = BlendSQL(
        {
            "w": pd.DataFrame(
                {
                    "description": [
                        "Go and buy AAPL",
                        "CSCO is not worth the buy",
                        "TYL isn't looking too good",
                    ]
                }
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
        SELECT * FROM w
        WHERE LENGTH(
            {{
                LLMMap(
                    'What stock ticker is mentioned in the text?',
                    description,
                    return_type='substring'
                )
            }}
        ) > 2 AND description NOT LIKE 'G%'
        """,
    )
    print(smoothie.df())
