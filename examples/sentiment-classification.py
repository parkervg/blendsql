import pandas as pd

from blendsql import BlendSQL
from blendsql.models import VLLM

if __name__ == "__main__":
    bsql = BlendSQL(
        {
            "posts": pd.DataFrame(
                {"content": ["I hate this product", "I love this product"]}
            )
        },
        model=VLLM(
            model_name_or_path="RedHatAI/gemma-3-12b-it-quantized.w4a16",
            base_url="http://127.0.0.1:8000/v1/",
        ),
        verbose=True,
    )

    smoothie = bsql.execute(
        """
        SELECT {{
            LLMMap(
                'What is the sentiment of this text?',
                content,
                options=('positive', 'negative')
            )      
        }} AS classification FROM posts LIMIT 10
        """
    )
    print(smoothie.df())
