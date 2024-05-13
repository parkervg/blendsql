## blend()

::: blendsql.blend.blend
    handler: python
    show_source: false

### Usage:

```python
from blendsql import blend, LLMMap, LLMQA, LLMJoin
from blendsql.db import SQLite
from blendsql.models import OpenaiLLM
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
smoothie = blend(
    query=blendsql,
    db=SQLite(fetch_from_hub("1884_New_Zealand_rugby_union_tour_of_New_South_Wales_1.db")),
    ingredients={LLMMap, LLMQA, LLMJoin},
    blender=OpenaiLLM("gpt-4"),
    # Optional args below
    infer_gen_constraints=True,
    silence_db_exec_errors=False,
    verbose=True,
    blender_args={
        "few_shot": True,
        "temperature": 0.01
    }
)
```

### Appendix

#### preprocess_blendsql()

::: blendsql.blend.preprocess_blendsql
    handler: python
    show_source: false