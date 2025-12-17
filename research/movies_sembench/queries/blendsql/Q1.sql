SELECT reviewId
FROM Reviews r
WHERE r.id = 'taken_3' AND {{
    LLMMap('Is the movie review clearly positive?', r.reviewText)
}} = TRUE
LIMIT 5;