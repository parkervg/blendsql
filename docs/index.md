<div align="right">
<a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" /></a>
<a><img src="https://img.shields.io/github/last-commit/parkervg/blendsql?color=green"/></a>
<a><img src="https://img.shields.io/badge/PRs-Welcome-Green"/></a>
<br>
</div>

<div align="center"><picture>
  <img alt="blendsql" src="img/logo_light.png" width=350">
</picture>
<p align="center">
    <i> SQL 🤝 LLMs </i>
  </p>
</div>
<br/>

## Intro
BlendSQL is a *superset of SQLite* for problem decomposition and hybrid question-answering with LLMs. It builds off of the syntax of SQL to create an intermediate representation for tasks requiring complex reasoning over both structured and unstructured data.

It can be viewed as an inversion of the typical text-to-SQL paradigm, where a user calls a LLM, and the LLM calls a SQL program.
Here, the user is given the control to oversee all calls (LLM + SQL) within a unified query language.

![comparison](img/
comparison.jpg)

For example, imagine we have the following tables.

### `w`
| **date** | **rival**                 | **city**  | **venue**                   | **score** |
|----------|---------------------------|-----------|-----------------------------|-----------|
| 31 may   | nsw waratahs              | sydney    | agricultural society ground | 11-0      |
| 5 jun    | northern districts        | newcastle | sports ground               | 29-0      |
| 7 jun    | nsw waratahs              | sydney    | agricultural society ground | 21-2      |
| 11 jun   | western districts         | bathurst  | bathurst ground             | 11-0      |
| 12 jun   | wallaroo & university nsw | sydney    | cricket ground              | 23-10     |

### `documents`
| **title**                      | **content**                                       |
|--------------------------------|---------------------------------------------------|
| sydney                         | sydney ( /ˈsɪdni/ ( listen ) sid-nee ) is the ... |
| new south wales waratahs       | the new south wales waratahs ( /ˈwɒrətɑːz/ or ... |
| sydney showground (moore park) | the former sydney showground ( moore park ) at... |
| sydney cricket ground          | the sydney cricket ground ( scg ) is a sports ... |
| newcastle, new south wales     | the newcastle ( /ˈnuːkɑːsəl/ new-kah-səl ) met... |
| bathurst, new south wales      | bathurst /ˈbæθərst/ is a city in the central t... |

BlendSQL allows us to ask the following questions by injecting "ingredients", which are callable functions denoted by double curly brackets (`{{`, `}}`).
The below examples work out of the box, but you are able to design your own ingredients as well! 

*What was the result of the game played 120 miles west of Sydney?*
```sql
SELECT * FROM w
    WHERE city = {{
        LLMQA(
            'Which city is located 120 miles west of Sydney?',
            (SELECT * FROM documents WHERE documents MATCH 'sydney OR 120'),
            options='w::city'
        )
    }}
```

*Which venues in Sydney saw more than 30 points scored?*
```sql
SELECT DISTINCT venue FROM w
    WHERE city = 'sydney' AND {{
        LLMMap(
            'More than 30 total points?',
            'w::score'
        )
    }} = TRUE
```

*Show all NSW Waratahs games and a description of the team.*
```sql
SELECT date, rival, score, documents.content AS "Team Description" FROM w
    JOIN {{
        LLMJoin(
            left_on='documents::title',
            right_on='w::rival'
        )
    }} WHERE rival = 'nsw waratahs'
```

### More Examples from Popular QA Datasets

<p>
<details>
<summary> <b> <a href="https://hybridqa.github.io/" target="_blank"> HybridQA </a> </b> </summary>

For this setting, our database contains 2 tables: a table from Wikipedia `w`, and a collection of unstructured Wikipedia articles in the table `documents`.

*What is the state flower of the smallest state by area ?*
```sql
SELECT "common name" AS 'State Flower' FROM w 
WHERE state = {{
    LLMQA(
        'Which is the smallest state by area?',
        (SELECT title, content FROM documents),
        options='w::state'
    )
}}
```

*Who were the builders of the mosque in Herat with fire temples ?*
```sql
{{
    LLMQA(
        'Name of the builders?',
        (
            SELECT title AS 'Building', content FROM documents
                WHERE title = {{
                    LLMQA(
                        'Align the name to the correct title.',
                        (SELECT name FROM w WHERE city = 'herat' AND remarks LIKE '%fire temple%'),
                        options='documents::title'
                    )
                }}
        ) 
    )
}}
```

*What is the capacity of the venue that was named in honor of Juan Antonio Samaranch in 2010 after his death ?*
```sql
SELECT capacity FROM w WHERE venue = {{
    LLMQA(
        'Which venue is named in honor of Juan Antonio Samaranch?',
        (SELECT title AS 'Venue', content FROM documents),
        options='w::venue'
    )
}}
```    

</details>
</p>

<p>
<details>
<summary> <b> <a href="https://ott-qa.github.io/" target="_blank"> OTT-QA </a> </b> </summary>

Unlike HybridQA, these questions are open-domain, where we don't know in advance where the answer of a given open question appears in a passage or a table.

As a result, we need to play the role of both the retriever (to select relevant context) and reader (to read from relevant contexts and return the given answer).

As the underlying database consists of 400K tables and 5M documents, it's important to set `LIMIT` clauses appropriately to ensure reasonable execution times.

The examples below also demonstrate how BlendSQL unpacks [CTE statements](https://www.sqlite.org/lang_with.html) to ensure we only pass necessary data into the BlendSQL ingredient calls. 

*When was the third highest paid Rangers F.C . player born ?*
```sql
{{
    LLMQA(
        'When was the Rangers Player born?'
        (
            WITH t AS (
                SELECT player FROM (
                    SELECT * FROM "./List of Rangers F.C. records and statistics (0)"
                    UNION ALL SELECT * FROM "./List of Rangers F.C. records and statistics (1)"
                ) ORDER BY trim(fee, '£') DESC LIMIT 1 OFFSET 2
            ), d AS (
                SELECT * FROM documents JOIN t WHERE documents MATCH t.player || ' OR rangers OR fc' ORDER BY rank LIMIT 5
            ) SELECT d.content, t.player AS 'Rangers Player' FROM d JOIN t
        )
    )
}}
```

*In which Track Cycling World Championships event was the person born in Matanzas , Cuba ranked highest ?*
```sql
{{
    LLMQA(
        'In what event was the cyclist ranked highest?',
        (
            SELECT * FROM (
                SELECT * FROM "./Cuba at the UCI Track Cycling World Championships (2)"
            ) as w WHERE w.name = {{
                LLMQA(
                    "Which cyclist was born in Matanzas, Cuba?",
                    (
                        SELECT * FROM documents 
                            WHERE documents MATCH 'matanzas AND (cycling OR track OR born)' 
                            ORDER BY rank LIMIT 3
                    ),
                    options="w::name"
                )
            }}
        ),
        options='w::event'
    )
}}
```

*Who is the director the Togolese film that was a 30 minute film that was shot in 16mm ?*
```sql
SELECT director FROM "./List of African films (4)" as w
WHERE title = {{
    LLMQA(
        'What is the name of the Togolese film that was 30 minutes and shot in 16mm?'
        (SELECT * FROM documents WHERE documents MATCH 'togolese OR 30 OR 16mm OR film' ORDER BY rank LIMIT 5)
        options='w::title'
    )
}}
```

</details>
</p>

<p>
<details>
<summary> <b> <a href="https://fever.ai/dataset/feverous.html" target="_blank"> FEVEROUS </a> </b> </summary>

Here, we deal not with questions, but truth claims given a context of unstructured and structured data.

These claims should be judged as "SUPPORTS" or "REFUTES". Using BlendSQL, we can formulate this determination of truth as a function over facts. 

*Oyedaea is part of the family Asteraceae in the order Asterales.*
```sql
SELECT EXISTS (
    SELECT * FROM w0 WHERE "family:" = 'asteraceae' AND "order:" = 'asterales'
) 
```

*The 2006-07 San Jose Sharks season, the 14th season of operation (13th season of play) for the National Hockey League (NHL) franchise, scored the most points in the Pacific Division.*
```sql
SELECT (
    {{
        LLMValidate(
            'Is the Sharks 2006-07 season the 14th season (13th season of play)?', 
            (SELECT * FROM documents)
        )
    }}
) AND (
    SELECT (SELECT filledcolumnname FROM w0 ORDER BY pts DESC LIMIT 1) = 'san jose sharks'
)
```

*Saunders College of Business, which is accredited by the Association to Advance Collegiate Schools of Business International, is one of the colleges of Rochester Institute of Technology established in 1910 and is currently under the supervision of Dean Jacqueline R. Mozrall.*
```sql
SELECT EXISTS(
    SELECT * FROM w0 
    WHERE "parent institution" = 'rochester institute of technology'
    AND "established" = '1910'
    AND "dean" = 'jacqueline r. mozrall'
) AND (
    {{
        LLMValidate(
            'Is Saunders College of Business (SCB) accredited by the Association to Advance Collegiate Schools of Business International (AACSB)?',
            (SELECT * FROM documents)
        )
    }}
)
```

</details>
</p>

### Features 
- Smart parsing optimizes what is passed to external functions 🧠
  - Traverses AST with [sqlglot](https://github.com/tobymao/sqlglot) to minimize external function calls
- Accelerated LLM calls, caching, and constrained decoding 🚀
  - Enabled via [guidance](https://github.com/guidance-ai/guidance)
- Easy logging of execution environment with `smoothie.save_recipe()` 🖥️
  - Enables reproducibility across machines


For a technical walkthrough of how a BlendSQL query is executed, check out [technical_walkthrough.md](./docs/technical_walkthrough.md).