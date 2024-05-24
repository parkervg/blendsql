---
hide:
  - toc
---
# Quickstart

```python
from blendsql import blend, LLMQA
from blendsql.db import SQLite
from blendsql.models import OpenaiLLM, TransformersLLM
from blendsql.utils import fetch_from_hub

blendsql = """
SELECT * FROM w
WHERE city = {{
    LLMQA(
        'Which city is located 120 miles west of Sydney?',
        (SELECT * FROM documents WHERE documents MATCH 'sydney OR 120'),
        options='w::city'
    )
}}
"""
# Make our smoothie - the executed BlendSQL script
smoothie = blend(
    query=blendsql,
    db=SQLite(fetch_from_hub("1884_New_Zealand_rugby_union_tour_of_New_South_Wales_1.db")),
    blender=OpenaiLLM("gpt-3.5-turbo"),
    # If you don't have OpenAI setup, you can use this small Transformers model below instead
    # blender=TransformersLLM("Qwen/Qwen1.5-0.5B"),
    ingredients={LLMQA},
)
print(smoothie.df)
print(smoothie.meta.prompts)
```