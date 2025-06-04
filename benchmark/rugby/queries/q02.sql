SELECT DISTINCT venue FROM w
    WHERE city = 'sydney' AND {{
        LLMMap(
            'More than 30 total points?',
            score
        )
    }} = TRUE