SELECT reviewId
FROM Reviews
WHERE id = 'taken_3' AND {{
    LLMMap(
        'Is the review sentiment clearly positive?',
        reviewText
    )
}} = TRUE
LIMIT 5
