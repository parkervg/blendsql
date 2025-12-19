SELECT reviewText, reviewId
FROM Reviews
WHERE {{
    LLMMap('Does this review have a positive sentiment?', reviewText)
}} = TRUE
LIMIT 5;