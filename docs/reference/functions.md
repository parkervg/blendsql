---
hide:
  - toc
---
# **General Syntax**
## `ValueArray`

A `ValueArray` is a reference to a list of values. This can be written using:

- Standard column `{tablename}.{columnname}` syntax (`tablename` can be ommitted, and standard SQL binding logic will apply)

- SQL tuple (`(value1, value2)`) syntax

- A BlendSQL query which returns a 1d array of values (`(SELECT value FROM table WHERE ...)`)

## Passing `options`

The functions `LLMMap` and `LLMQA` support the passing of an `options` argument. This will constrain the output of the functions to only values appearing in the passed [`ValueArray`](#`valuearray`).

```sql
SELECT {{
    LLMMap(
        'What is the sentiment of this text?',
        content,
        options=('positive', 'negative')
    )      
}}, content AS classification FROM posts LIMIT 10
```

### `Quantifier`

An optional `quantifier` argument can be passed to `LLMQA` and `LLMMap`, which will be used to modify the regular expression pattern powering the constrained decoding. For example:

```sql
SELECT {{
    LLMMap(
        'What are their interests?',
        context_column,
        quantifier='+'
    )
}} FROM People
```

Since we're asking for 'one-or-more' via the quantifier arg and the default `return_type` (as of v0.0.61) is `str`, BlendSQL casts the `return_type` to a `List[str]`. 

The following [greedy quantifiers](https://learn.microsoft.com/en-us/dotnet/standard/base-types/quantifiers-in-regular-expressions) are valid:

- `'*'`, meaning 'zero-or-more'
- `'+`', meaning 'one-or-more'
- Any string matching the pattern `{\d(,\d)?}` (e.g. `{1,2}`)
- 
# **Functions**
## LLMQA

The `LLMQA` is an aggregate function that returns a single scalar value.

```python
def LLMQA(
    question: str,
    *context: Query,
    options: Optional[ValueArray] = None,
    return_type: Optional[ReturnType] = None,
    regex: Optional[str] = None,
    quantifier: Optional[Quantifier] = None
):
    ...
```

Examples:
```sql
SELECT preferred_foot FROM Player p
WHERE p.player_name = {{
    /* With `infer_gen_constraints=True` (which is default),
    `options` will automatically be inferred, and the below
    will select from a value in the `p.player_name` column. */
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
        /* Pass context via a subquery */
        (SELECT title, content FROM documents)
    )
}}
```

```sql
/* Generate 3 values in our generated tuple */
SELECT * FROM VALUES {{LLMQA('What are the first 3 letters of the alphabet?', quantifier='{3}')}}
```

```sql
SELECT {{
    LLMQA(
        /* Use f-string templating to insert the result of subqueries*/
        'What do {} and {} have in common?',
        /* Below are examples - any BlendSQL queries are valid here, 
        but they should return a single scalar value.   
        */
        (SELECT 'Saturn'),
        (SELECT 'Jupiter')
    )    
}}
```

#### Also See:
- [LLMQA with search](https://github.com/parkervg/blendsql/blob/main/examples/vector-search-reduce.py)
- [LLMQA with search + f-string templating](https://github.com/parkervg/blendsql/blob/main/examples/llmqa-f-string.py#L9)

## LLMMap

The `LLMMap` is a unary scalar function, much like `LENGTH` or `ABS` in SQlite. The output of this function is set as a new column in a temporary table, for later use within the wider query.

```python
def LLMMap(
    question: str,
    values: ColumnRef,
    *context: ValueArray,
    options: Optional[ValueArray] = None,
    return_type: Optional[ReturnType] = None,
    regex: Optional[str] = None
):
    ...
```

Examples:
```sql
SELECT COUNT(DISTINCT(s.CDSCode)) FROM schools s
JOIN satscores sa ON s.CDSCode = sa.cds
WHERE sa.AvgScrMath > 560
/* With `infer_gen_constraints=True`, generations below will be restricted to a boolean. */
AND {{LLMMap('Is this a county in the California Bay Area?', s.County)}} = TRUE
```

```sql
SELECT GROUP_CONCAT(Name, ', ') AS Names,
{{
    LLMMap(
        'In which time period was this person born?',
        p.Name,
        /* BlendSQL differs from standard SQL binding logic below, 
        since we can invoke a table (`Eras`) not previously referenced */
        options=Eras.Years
    )
}} AS Born
FROM People p
GROUP BY Born
```

```sql
WITH player_stats AS (
    SELECT *, {{
        LLMMap(
            'How many points and assists did {} have? Respond in the order [points, assists]. If a stat is not present for a player, return -1.', 
            player, 
            Report, /* Pass `Report` in as context for each `player` */
            return_type='List[int]',
            quantifier='{2}'
        )
    }} AS box_score_values
    FROM w
) SELECT 
player,
Report,
list_element(box_score_values, 1) AS points,
list_element(box_score_values, 2) AS assists
FROM player_stats
```

#### Also See:
- [LLMMap with search](https://github.com/parkervg/blendsql/blob/main/examples/vector-search-map.py)
- [Mapping with `return_type='substring'`](https://github.com/parkervg/blendsql/blob/main/examples/map-substring.py)

## LLMJoin

The `LLMJoin` function can be used to perform semantic entity linking between columns in tables. It is commonly used in conjunction with a `documents` table, to fetch articles related to a value in another table.

```python
def LLMJoin(
    left_on: ValueArray,
    right_on: ValueArray,
    join_criteria: Optional[str] = "Join to same topics."
):
    ...
```

Examples:
```sql
-- Get all articles on players older than 21
SELECT * FROM Player p
JOIN documents d ON {{
    LLMJoin(
        p.Name,
        d.title
    )
}} WHERE p.age > 21
```

```sql
SELECT f.name, c.name FROM fruits f
JOIN colors c ON {{
    LLMJoin(
        f.name,
        c.name,
        /* If we need to, we can pass a join_criteria.
        Otherwise, the default 'Join by topic' is used. */
        join_criteria='Align the fruit to its color.'
    )
}}
```
