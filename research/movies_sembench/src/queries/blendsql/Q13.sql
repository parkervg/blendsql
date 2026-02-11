SELECT reviewId
FROM Reviews
WHERE {{
    LLMMap(
        'Does the review have a positive sentiment?',
        reviewText
    )
}} = TRUE