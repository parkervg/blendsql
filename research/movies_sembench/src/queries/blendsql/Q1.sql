SELECT reviewId
FROM Reviews
WHERE {{
    LLMMap(
        'Is the review sentiment positive?',
        reviewText
    )
}} = TRUE
LIMIT 5;