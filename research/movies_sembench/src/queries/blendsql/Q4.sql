SELECT SUM(
    {{
        LLMMap(
            'Does the movie review have a positive sentiment?',
            reviewText
        )
    }} = TRUE
) / COUNT(*) AS positive_reviews_ratio
FROM Reviews
WHERE id = 'taken_3'