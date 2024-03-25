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