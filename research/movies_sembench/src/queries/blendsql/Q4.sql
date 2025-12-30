SELECT SUM(
    {{
        LLMMap(
            'Is the review sentiment clearly positive?',
            reviewText
        )
    }} = TRUE
) / COUNT(*) AS positive_reviews_ratio
FROM Reviews
WHERE id = 'taken_3'