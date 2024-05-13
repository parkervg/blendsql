# [HybridQA](https://hybridqa.github.io/)

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
        'Who were the builders of the mosque?',
        (
            SELECT documents.title AS 'Building', documents.content FROM documents
            JOIN {{
                LLMJoin(
                    left_on='w::name',
                    right_on='documents::title'
                )
            }}
            WHERE w.city = 'herat' AND w.remarks LIKE '%fire temple%'
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