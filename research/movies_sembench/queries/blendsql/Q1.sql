SELECT reviewId
FROM Reviews r
WHERE {{
    LLMMap('Is the movie review clearly positive?', r.reviewText)
}} = TRUE
LIMIT 5;