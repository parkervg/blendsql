SELECT SUM(
    {{
        LLMMap(
            'Is this a positive review?',
            reviewText
        )
    }} = TRUE
) / COUNT(*) AS positive_reviews_ratio
FROM Reviews
WHERE id = 'taken_3'