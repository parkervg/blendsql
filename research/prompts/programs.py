# Define our BlendSQL prompt
# If the questions comment on some relative datetime operation, use the new grammar:
#     `DT('{table}::{column}', start='', end='')`
fewshot_blendsql_program_chat = """
{{#system~}}
Generate BlendSQL given the question to answer the question correctly.
BlendSQL is a superset of SQLite, which adds external function calls for information not found within native SQLite.
These external functions should be wrapped in double curly brackets.

If question-relevant column(s) contents are not suitable for SQL comparisons or calculations, map it to a new column with clean content by a new grammar:
    `LLMMap('question', '{table}::{column}')`

If mapping to a new column still cannot answer the question with valid SQL, turn to an end-to-end solution using a new grammar:
    `LLMQA('{question}', ({blendsql}))`
    
If we need to do a `join` operation where there is imperfect alignment between table values, use the new grammar:
    `LLMJoin(({blendsql}), options='{table}::{column}')`

ONLY use these BlendSQL functions if necessary. 
Answer parts of the question in vanilla SQL, if possible.

{{#if extra_task_description}}
{{extra_task_description}}
{{/if}}
{{~/system}}

{{#user~}}
Examples:
{{~#each examples}}
{{this.serialized_db}}
Question: {{this.question}}
BlendSQL: {{this.blendsql}}
{{/each}}

{{serialized_db}}
{{#if bridge_hints}}
Here are some values that may be helpful:
{{bridge_hints}}
{{/if}}
Question: {{question}}
BlendSQL:
{{~/user}}

{{#assistant~}}
{{gen "result" temperature=0.0}}
{{~/assistant}}
"""

fewshot_blendsql_program_completion = """
Generate BlendSQL given the question to answer the question correctly.
BlendSQL is a superset of SQLite, which adds external function calls for information not found within native SQLite.
These external functions should be wrapped in double curly brackets.

If question-relevant column(s) contents are not suitable for SQL comparisons or calculations, map it to a new column with clean content by a new grammar:
    `LLMMap('question', '{table}::{column})'`
    
If mapping to a new column still cannot answer the question with valid SQL, turn to an end-to-end solution using a new grammar:
    `LLMQA('{question}', ({blendsql}))`

ONLY use these BlendSQL functions if necessary. 
Answer parts of the question in vanilla SQL, if possible.

{{#if extra_task_description}}
{{extra_task_description}}
{{/if}}

Examples:
{{~#each examples}}
{{this.serialized_db}}
Question: {{this.question}}
BlendSQL: {{this.blendsql}}
{{/each}}

{{serialized_db}}
{{#if bridge_hints}}
Here are some values that may be helpful:
{{bridge_hints}}
{{/if}}
Question: {{question}}
BlendSQL: {{gen "result" temperature=0.0}}
"""

fewshot_sql_program_chat = """
{{#system~}}
Generate SQL given the question and table to answer the question correctly.
{{#if extra_task_description}}
{{extra_task_description}}
{{/if}}

Examples:
{{~#each examples}}
{{this.serialized_db}}
Question: {{this.question}}
SQL: {{this.sql}}
{{/each}}

{{~/system}}

{{#user~}}
{{serialized_db}}

{{#if bridge_hints}}
Here are some values that may be helpful:
{{bridge_hints}}
{{/if}}

Question: {{question}}
SQL:
{{~/user}}

{{#assistant~}}
{{gen "result" temperature=0.0}}
{{~/assistant}}
"""

fewshot_sql_program_completion = """
Generate SQL given the question and table to answer the question correctly.
{{#if extra_task_description}}
{{extra_task_description}}
{{/if}}
{{~/system}}

Examples:
{{~#each examples}}
{{this.serialized_db}}
Question: {{this.question}}
SQL: {{this.sql}}
{{/each}}

{{serialized_db}}

{{#if bridge_hints}}
Here are some values that may be helpful:
{{bridge_hints}}
{{/if}}

Question: {{question}}
SQL: {{gen "result" temperature=0.0}}
"""

zero_shot_qa_program_chat = """
{{#system~}}
This is a hybrid question answering task. The goal of this task is to answer the question given a table (`w`) and corresponding passages (`docs`).
Be as succinct as possible in answering the given question, do not include explanation.
{{~/system}}

{{#user~}}
Context:
{{serialized_db}}

Question: {{question}}
Answer:
{{~/user}}

{{#assistant~}}
{{gen "result" temperature=0.0 max_tokens=50}}
{{~/assistant}}
"""
