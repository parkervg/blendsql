# [OTT-QA](https://ott-qa.github.io/)

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