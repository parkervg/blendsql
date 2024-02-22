# Quickstart

```python
from blendsql import blend, LLMQA, LLMMap
from blendsql.db import SQLiteDBConnector
from blendsql.llms import OpenaiLLM
from blendsql.utils import fetch_from_hub

blendsql = """
SELECT merchant FROM transactions WHERE 
     {{LLMMap('is this an italian restaurant?', 'transactions::merchant')}} = TRUE
     AND parent_category = 'Food'
"""
# Make our smoothie - the executed BlendSQL script
smoothie = blend(
    query=blendsql,
    blender=OpenaiLLM("gpt-3.5-turbo-0613"),
    ingredients={LLMMap, LLMQA},
    db=SQLiteDBConnector(fetch_from_hub("single_table.db")),
    verbose=True
)

```