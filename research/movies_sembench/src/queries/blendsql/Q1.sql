SELECT reviewId
FROM Reviews
WHERE {{
    LLMMap(
        'Is the movie review clearly positive?',
        reviewText
    )
}} = TRUE
LIMIT 5;