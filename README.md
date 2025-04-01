<div align="right">
<a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" /></a>
<a><img src="https://img.shields.io/github/last-commit/parkervg/blendsql?color=green"/></a>
<a><img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/parkervg/e24f1214fdff3ab086b829b5f01f85a8/raw/covbadge.json"/></a>
<a><img src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue"/></a>
<br>
</div>

<div align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/logo_dark.png">
  <img alt="blendsql" src="docs/img/logo_light.png" width=350">
</picture>
<p align="center">
    <i> SQL 🤝 LLMs </i>
  </p>
<b><h3>Check out our <a href="https://parkervg.github.io/blendsql/" target="_blank">online documentation</a> for a more comprehensive overview.</h3></b>

</div>
<br/>

```
pip install blendsql
```

```python
import pandas as pd

from blendsql import BlendSQL
from blendsql.ingredients import LLMMap, LLMQA, LLMJoin
from blendsql.models import TransformersLLM

# Load model
model = TransformersLLM(
   "meta-llama/Llama-3.2-1B-Instruct"
) # Local models enable BlendSQL's predicate-guided constrained decoding

# Prepare our BlendSQL connection
bsql = BlendSQL(
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
                    "Elvis Presley",
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
                    "14 Grammys, King of Rock n Roll",
                ],
            }
        ),
        "Eras": pd.DataFrame({"Years": ["1800-1900", "1900-2000", "2000-Now"]}),
    },
    ingredients={LLMMap, LLMQA, LLMJoin},
    model=model,
    verbose=True
)

smoothie = bsql.execute(
    """
    SELECT * FROM People P
    WHERE P.Name IN {{
        LLMQA('First 3 presidents of the U.S?', modifier='{3}')
    }}
    """,
    infer_gen_constraints=True
)

print(smoothie.df)
# ┌───────────────────┬───────────────────────────────────────────────────────┐
# │ Name              │ Known_For                                             │
# ├───────────────────┼───────────────────────────────────────────────────────┤
# │ George Washington │ Established federal government, First U.S. Preside... │
# │ John Quincy Adams │ XYZ Affair, Alien and Sedition Acts                   │
# │ Thomas Jefferson  │ Louisiana Purchase, Declaration of Independence       │
# └───────────────────┴───────────────────────────────────────────────────────┘
print(smoothie.summary())
# ┌────────────┬──────────────────────┬─────────────────┬─────────────────────┐
# │   Time (s) │   # Generation Calls │   Prompt Tokens │   Completion Tokens │
# ├────────────┼──────────────────────┼─────────────────┼─────────────────────┤
# │    1.25158 │                    1 │             296 │                  16 │
# └────────────┴──────────────────────┴─────────────────┴─────────────────────┘
```

### ✨ News
- (3/16/25) Use BlendSQL with 100+ LLM APIs, using [LiteLLM](https://github.com/BerriAI/litellm)!
- (10/26/24) New tutorial! [blendsql-by-example.ipynb](examples/blendsql-by-example.ipynb)
- (10/18/24) Concurrent async requests in 0.0.29! OpenAI and Anthropic `LLMMap` calls are speedy now.
  - Customize max concurrent async calls via `blendsql.config.set_async_limit(10)`
- (10/15/24) As of version 0.0.27, there is a new pattern for defining + retrieving few-shot prompts; check out [Few-Shot Prompting](#few-shot-prompting) in the README for more info
- (10/15/24) Check out [Some Cool Things by Example](https://parkervg.github.io/blendsql/by-example/) for some recent language updates!

BlendSQL is a *superset of SQL* for problem decomposition and hybrid question-answering with LLMs.

As a result, we can *Blend* together...

- 🥤 ...operations over heterogeneous data sources (e.g. tables, text, images)
- 🥤 ...the structured & interpretable reasoning of SQL with the generalizable reasoning of LLMs


**Now, the user is given the control to oversee all calls (LLM + SQL) within a unified query language.**

### Features

- Supports many DBMS 💾
  - SQLite, PostgreSQL, DuckDB, Pandas (aka duckdb in a trenchcoat)
- Supports many models ✨
  - Transformers, OpenAI, Anthropic, Ollama
- Easily extendable to [multi-modal usecases](./examples/vqa-ingredient.ipynb) 🖼️
- Write your normal queries - smart parsing optimizes what is passed to external functions 🧠
  - Traverses abstract syntax tree with [sqlglot](https://github.com/tobymao/sqlglot) to minimize LLM function calls 🌳
- Constrained decoding with [guidance](https://github.com/guidance-ai/guidance) 🚀
  - When using local models, we only generate syntactically valid outputs according to query syntax + database contents
- LLM function caching, built on [diskcache](https://grantjenks.com/docs/diskcache/) 🔑


![comparison](docs/img/comparison.jpg)

### Citation

```bibtex
@article{glenn2024blendsql,
      title={BlendSQL: A Scalable Dialect for Unifying Hybrid Question Answering in Relational Algebra},
      author={Parker Glenn and Parag Pravin Dakle and Liang Wang and Preethi Raghavan},
      year={2024},
      eprint={2402.17882},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Few-Shot Prompting
For the LLM-based ingredients in BlendSQL, few-shot prompting can be vital. In `LLMMap`, `LLMQA` and `LLMJoin`, we provide an interface to pass custom few-shot examples and dynamically retrieve those top-`k` most relevant examples at runtime, given the current inference example.
#### `LLMMap`
- [Default examples](./blendsql/ingredients/builtin/map/default_examples.json)
- [All possible fields](./blendsql/ingredients/builtin/map/examples.py)

```python
from blendsql import BlendSQL
from blendsql.ingredients.builtin import LLMMap, DEFAULT_MAP_FEW_SHOT

ingredients = {
    LLMMap.from_args(
        few_shot_examples=[
            *DEFAULT_MAP_FEW_SHOT,
            {
                "question": "Is this a sport?",
                "mapping": {
                    "Soccer": "t",
                    "Chair": "f",
                    "Banana": "f",
                    "Golf": "t"
                },
                # Below are optional
                "column_name": "Items",
                "table_name": "Table",
                "example_outputs": ["t", "f"],
                "options": ["t", "f"],
                "output_type": "boolean"
            }
        ],
        # Will fetch `k` most relevant few-shot examples using embedding-based retriever
        k=2,
        # How many inference values to pass to model at once
        batch_size=5,
    )
}

bsql = BlendSQL(db, ingredients=ingredients)
```

#### `LLMQA`
- [Default examples](./blendsql/ingredients/builtin/qa/default_examples.json)
- [All possible fields](./blendsql/ingredients/builtin/qa/examples.py)

```python
from blendsql import BlendSQL
from blendsql.ingredients.builtin import LLMQA, DEFAULT_QA_FEW_SHOT

ingredients = {
    LLMQA.from_args(
        few_shot_examples=[
            *DEFAULT_QA_FEW_SHOT,
            {
                "question": "Which weighs the most?",
                "context": {
                    {
                        "Animal": ["Dog", "Gorilla", "Hamster"],
                        "Weight": ["20 pounds", "350 lbs", "100 grams"]
                    }
                },
                "answer": "Gorilla",
                # Below are optional
                "options": ["Dog", "Gorilla", "Hamster"]
            }
        ],
        # Will fetch `k` most relevant few-shot examples using embedding-based retriever
        k=2,
        # Lambda to turn the pd.DataFrame to a serialized string
        context_formatter=lambda df: df.to_markdown(
            index=False
        )
    )
}

bsql = BlendSQL(db, ingredients=ingredients)
```

#### `LLMJoin`
- [Default examples](./blendsql/ingredients/builtin/join/default_examples.json)
- [All possible fields](./blendsql/ingredients/builtin/join/examples.py)

```python
from blendsql import BlendSQL
from blendsql.ingredients.builtin import LLMJoin, DEFAULT_JOIN_FEW_SHOT

ingredients = {
    LLMJoin.from_args(
        few_shot_examples=[
            *DEFAULT_JOIN_FEW_SHOT,
            {
                "join_criteria": "Join the state to its capital.",
                "left_values": ["California", "Massachusetts", "North Carolina"],
                "right_values": ["Sacramento", "Boston", "Chicago"],
                "mapping": {
                    "California": "Sacramento",
                    "Massachusetts": "Boston",
                    "North Carolina": "-"
                }
            }
        ],
        # Will fetch `k` most relevant few-shot examples using embedding-based retriever
        k=2
    )
}

bsql = BlendSQL(db, ingredients=ingredients)
```


### Acknowledgements
Special thanks to those below for inspiring this project. Definitely recommend checking out the linked work below, and citing when applicable!

- The authors of [Binding Language Models in Symbolic Languages](https://arxiv.org/abs/2210.02875)
  - This paper was the primary inspiration for BlendSQL.
- The authors of [EHRXQA: A Multi-Modal Question Answering Dataset for Electronic Health Records with Chest X-ray Images](https://arxiv.org/pdf/2310.18652)
  - As far as I can tell, the first publication to propose unifying model calls within SQL
  - Served as the inspiration for the [vqa-ingredient.ipynb](./examples/vqa-ingredient.ipynb) example
- The authors of [Grammar Prompting for Domain-Specific Language Generation with Large Language Models](https://arxiv.org/abs/2305.19234)
- The maintainers of the [Guidance](https://github.com/guidance-ai/guidance) library for powering the constrained decoding capabilities of BlendSQL
