## Execute a BlendSQL Query
::: blendsql.blendsql.blend
    handler: python
    show_source: false

### Usage:
```python
from blendsql import blend, LLMMap, LLMQA, LLMJoin
from blendsql.db import SQLiteDBConnector
from blendsql.llms import OpenaiLLM

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
db = SQLiteDBConnector(db_path)
smoothie = blend(
    query=blendsql,
    db=db,
    ingredients={LLMMap, LLMQA, LLMJoin},
    blender=AzureOpenaiLLM("gpt-4"),
    # Optional args below
    infer_map_constraints=True,
    silence_db_exec_errors=False,
    verbose=True,
    blender_args={
      "few_shot": True,
      "temperature": 0.01
    }
)
```