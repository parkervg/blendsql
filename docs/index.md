---
hide:
  - toc
---
<div align="right">
<a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" /></a>
<a><img src="https://img.shields.io/github/last-commit/parkervg/blendsql?color=green"/></a>
<a><img src="https://img.shields.io/badge/PRs-Welcome-Green"/></a>
<br>
</div>

<center>
<picture> 
  <img alt="blendsql" src="img/logo_light.png" width=350">
</picture>
<br>
    <i> SQL 🤝 LLMs </i>
<br><br>
[Paper :simple-arxiv:](https://arxiv.org/pdf/2402.17882.pdf){ .md-button } [GitHub :simple-github:](https://github.com/parkervg/blendsql){ .md-button }

<div class="index-pre-code">
```bash
pip install blendsql
```
</div>
</center>

BlendSQL is a *superset of SQLite* for problem decomposition and hybrid question-answering with LLMs. It builds off of the syntax of SQL to create an intermediate representation for tasks requiring complex reasoning over both structured and unstructured data.

It can be viewed as an inversion of the typical text-to-SQL paradigm, where a user calls a LLM, and the LLM calls a SQL program.
Here, the user is given the control to oversee all calls (LLM + SQL) within a unified query language.

As shown in our paper, using BlendSQL as an intermediate representation for complex reasoning tasks can <b>boost performance by 8.63%</b> and <b>use 35% fewer tokens</b> compared to the naive end-to-end approach. For an example of this approach, see [this notebook](reference/examples/teaching-blendsql-via-in-context-learning).

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

### Features 
- Smart parsing optimizes what is passed to external functions 🧠
  - Traverses abstract syntax tree with [sqlglot](https://github.com/tobymao/sqlglot) to minimize LLM function calls 🌳
- LLM function caching, built on [diskcache](https://grantjenks.com/docs/diskcache/) 🔑 
- Constrained decoding with [guidance](https://github.com/guidance-ai/guidance) 🚀


For a technical walkthrough of how a BlendSQL query is executed, check out [technical_walkthrough.md](reference/technical_walkthrough.md).

### Citation

```
@article{glenn2024blendsql,
      title={BlendSQL: A Scalable Dialect for Unifying Hybrid Question Answering in Relational Algebra}, 
      author={Parker Glenn and Parag Pravin Dakle and Liang Wang and Preethi Raghavan},
      year={2024},
      eprint={2402.17882},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```