SELECT * FROM w
    WHERE city = {{
        LLMQA(
            'Which city is located 120 miles west of Sydney?',
            (SELECT * FROM documents WHERE documents MATCH 'sydney OR 120'),
            options=city
        )
    }}