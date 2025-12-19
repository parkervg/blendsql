SELECT reviewId
FROM Reviews
WHERE id = 'taken_3' AND {{
    LLMMap('Does this review have a positive sentiment?', reviewText)
}} = TRUE
LIMIT 5;
