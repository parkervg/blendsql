SELECT reviewId
FROM Reviews
WHERE {{
    LLMMap(
        'Is the review sentiment clearly positive?',
        reviewText
    )
}} = TRUE
LIMIT 5;