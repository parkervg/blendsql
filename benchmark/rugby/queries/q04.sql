SELECT rival
FROM w
WHERE city = {{
    LLMQA(
        'What city features the Mount Panorama racetrack?',
        (SELECT title, content FROM documents WHERE documents MATCH 'mount panorama racetrack'),
        options='w::city'
    )
}}