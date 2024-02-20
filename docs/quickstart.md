# Quickstart

```python
from blendsql import blend, LLMQA, LLMMap
from blendsql.db import SQLiteDBConnector
from blendsql.llms import OpenaiLLM

blendsql = """
SELECT merchant FROM transactions WHERE 
     {{LLMMap('is this a pizza shop?', 'transactions::merchant'}} = TRUE
     AND parent_category = 'Food'
"""
# Make our smoothie - the executed BlendSQL script
smoothie = blend(
    query=blendsql,
    blender=OpenaiLLM("gpt-3.5-turbo-0613"),
    ingredients={LLMMap, LLMQA},
    db=SQLiteDBConnector(db_path="transactions.db"),
    verbose=True
)

```