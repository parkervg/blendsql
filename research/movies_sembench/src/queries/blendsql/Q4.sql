SELECT SUM(
    {{
        LLMMap(
            'What is the sentiment of this review?',
            reviewText,
            options=('POSITIVE', 'NEGATIVE')
        )
    }} = 'POSITIVE'
) / COUNT(*) AS positive_reviews_ratio
FROM Reviews
WHERE id = 'taken_3'