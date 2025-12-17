SELECT SUM(
    {{
        LLMMap(
            'Is this a positive review?',
            r.reviewText
        )
    }} = TRUE
) / COUNT(*) AS positive_reviews_ratio
FROM Reviews AS r
WHERE id = 'taken_3'