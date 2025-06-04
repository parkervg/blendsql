import pandas as pd
import torch.cuda

from blendsql import BlendSQL
from blendsql.models import LiteLLM, LlamaCpp

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
        model=LlamaCpp(
            "Meta-Llama-3.1-8B-Instruct.Q6_K.gguf",
            "QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF",
            config={"n_gpu_layers": -1},
        )
        if torch.cuda.is_available()
        else LiteLLM("openai/gpt-4o"),
        verbose=True,
    )
    smoothie = bsql.execute(
        """
        SELECT * FROM fruit f
        JOIN colors c ON {{
            LLMJoin(
                f.name,
                c.name
            )
        }}
        """,
    )
