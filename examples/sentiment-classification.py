import pandas as pd

from blendsql import BlendSQL
from blendsql.models import LiteLLM

if __name__ == "__main__":
    bsql = BlendSQL(
        {
            "posts": pd.DataFrame(
                {"content": ["I hate this product", "I love this product"]}
            )
        },
        model=LiteLLM("openai/gpt-4o", caching=False),
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
    print(smoothie.df)
