import pandas as pd
import torch.cuda

from blendsql import BlendSQL
from blendsql.models import LlamaCpp, TransformersLLM

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
        model=LlamaCpp(
            "Meta-Llama-3.1-8B-Instruct.Q6_K.gguf",
            "QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF",
            config={"n_gpu_layers": -1},
        )
        if torch.cuda.is_available()
        # else LiteLLM("openai/gpt-4o"),
        else TransformersLLM("HuggingFaceTB/SmolLM2-135M-Instruct"),
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
    print(smoothie.df)
