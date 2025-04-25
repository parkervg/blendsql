---
hide:
  - toc
---
# **General Syntax**
## `ValueArray`

A `ValueArray` is a reference to a list of values. This can be written using:

- `'{value1};{value2}...'` syntax

- `'{table}::{column}'` syntax

- A BlendSQL query which returns a 1d array of values (`(SELECT value FROM table WHERE ...)`)

## Passing `options`

The functions `LLMMap` and `LLMQA` support the passing of an `options` argument. This will constrain the output of the functions to only values appearing in the passed [`ValueArray`](#`valuearray`).

```sql
SELECT Id FROM posts p
WHERE {{
    LLMMap(
        'What is the sentiment of this post?',
        'p::Body',
        options='positive;negative'
}}
```

# **Functions**
## LLMQA

The `LLMQA` is an aggregate function that returns a single scalar value.

### `Modifier`

An optional `modifier` argument can be passed, which will be used to modify the regular expression pattern powering the constrained decoding. The following are valid:

- `'*'`, meaning 'zero-or-more'
- `'+`', meaning 'one-or-more'
- Any string matching the pattern `{\d(,\d)?}`

```python
def LLMQA(
    question: str,
    context: t.Optional[Query] = None,
    options: t.Optional[ValueArray] = None,
    output_type: t.Optional[OutputType] = None,
    regex: t.Optional[str] = None,
    modifier: t.Optional[Modifier] = None
):
    ...
```

Examples:
```sql
SELECT preferred_foot FROM Player p
WHERE p.player_name = {{
    -- With `infer_gen_constraints=True` (which is default),
    --  `options` will automatically be inferred, and the below
    --  will select from a value in the `p.player_name` column.
    LLMQA(
        "Which player has the most Ballon d'Or awards?"
    )
}}
```

```sql
SELECT name FROM state_flowers
WHERE state = {{
    LLMQA(
        "Which state is known as 'The Golden State'?",
        -- Pass context via a subquery
        (SELECT title, content FROM documents)
    )
}}
```

```sql
-- Generate 3 values in our generated tuple
SELECT * FROM VALUES {{LLMQA('What are the first 3 letters of the alphabet?', modifier='{3}')}}
```

## LLMMap

The `LLMMap` is a unary scalar function, much like `LENGTH` or `ABS` in SQlite. The output of this function is set as a new column in a temporary table, for later use within the wider query.

```python
def LLMMap(
    question: str,
    values: ValueArray,
    options: t.Optional[ValueArray] = None,
    output_type: t.Optional[OutputType] = None,
    regex: t.Optional[str] = None
):
    ...
```

Examples:
```sql
SELECT COUNT(DISTINCT(s.CDSCode)) FROM schools s
JOIN satscores sa ON s.CDSCode = sa.cds
WHERE sa.AvgScrMath > 560
-- With `infer_gen_constraints=True`, generations below will be restricted to a boolean.
AND {{LLMMap('Is this a county in the California Bay Area?', 's::County')}} = TRUE
```

```sql
SELECT GROUP_CONCAT(Name, ', ') AS Names,
{{
    LLMMap(
        'In which time period was this person born?',
        'People::Name',
        options='Eras::Years'
    )
}} AS Born
FROM People
GROUP BY Born
```

## LLMJoin

The `LLMJoin` function can be used to perform semantic entity linking between columns in tables. It is commonly used in conjunction with a `documents` table, to fetch articles related to a value in another table.

```python
def LLMJoin(
    left_on: ValueArray,
    right_on: ValueArray,
    join_criteria: t.Optional[str]
):
    ...
```

Examples:
```sql
-- Get all articles on players older than 21
SELECT * FROM Player p
JOIN {{
    LLMJoin(
        'p::Name',
        'documents::title'
    )
}} WHERE p.age > 21
```

```sql
SELECT fruits.name, colors.name FROM fruits
JOIN {{
    LLMJoin(
        'fruits::name',
        'colors::name',
        -- If we need to, we can pass a join_criteria.
        -- Otherwise, the default 'Join by topic' is used.
        'Align the fruit to its color.'
    )
}}
```
