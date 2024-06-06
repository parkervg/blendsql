---
hide:
  - toc
---
# Quickstart

```python
import pandas as pd

from blendsql import blend, LLMMap, LLMQA, LLMJoin
from blendsql.db import Pandas
from blendsql.models import TransformersLLM

# Load model
model = TransformersLLM('Qwen/Qwen1.5-0.5B')

# Prepare our local database
db = Pandas(
    {
        "w": pd.DataFrame(
            (
                ['11 jun', 'western districts', 'bathurst', 'bathurst ground', '11-0'],
                ['12 jun', 'wallaroo & university nsq', 'sydney', 'cricket ground',
                 '23-10'],
                ['5 jun', 'northern districts', 'newcastle', 'sports ground', '29-0']
            ),
            columns=['date', 'rival', 'city', 'venue', 'score']
        ),
        "documents": pd.DataFrame(
            (
                ['bathurst, new south wales', 'bathurst /ˈbæθərst/ is a city in the central tablelands of new south wales , australia . it is about 200 kilometres ( 120 mi ) west-northwest of sydney and is the seat of the bathurst regional council .'],
                ['sydney', 'sydney ( /ˈsɪdni/ ( listen ) sid-nee ) is the state capital of new south wales and the most populous city in australia and oceania . located on australia s east coast , the metropolis surrounds port jackson.'],
                ['newcastle, new south wales', 'the newcastle ( /ˈnuːkɑːsəl/ new-kah-səl ) metropolitan area is the second most populated area in the australian state of new south wales and includes the newcastle and lake macquarie local government areas .']
            ),
            columns=['title', 'content']
        )
    }
)

# Write BlendSQL query
blendsql = """
SELECT * FROM w
WHERE city = {{
    LLMQA(
        'Which city is located 120 miles west of Sydney?',
        (SELECT * FROM documents WHERE content LIKE '%sydney%'),
        options='w::city'
    )
}}
"""
smoothie = blend(
    query=blendsql,
    db=db,
    ingredients={LLMMap, LLMQA, LLMJoin},
    blender=model,
    # Optional args below
    infer_gen_constraints=True,
    verbose=True
)
print(smoothie.df)
# ┌────────┬───────────────────┬──────────┬─────────────────┬─────────┐
# │ date   │ rival             │ city     │ venue           │ score   │
# ├────────┼───────────────────┼──────────┼─────────────────┼─────────┤
# │ 11 jun │ western districts │ bathurst │ bathurst ground │ 11-0    │
# └────────┴───────────────────┴──────────┴─────────────────┴─────────┘
print(smoothie.meta.prompts)
# [
#   {
#       'answer': 'bathurst',
#       'question': 'Which city is located 120 miles west of Sydney?',
#       'context': [
#           {'title': 'bathurst, new south wales', 'content': 'bathurst /ˈbæθərst/ is a city in the central tablelands of new south wales , australia . it is about...'},
#           {'title': 'sydney', 'content': 'sydney ( /ˈsɪdni/ ( listen ) sid-nee ) is the state capital of new south wales and the most populous city in...'}
#       ]
#    }
# ]
```