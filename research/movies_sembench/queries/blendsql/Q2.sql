SELECT reviewId
FROM Reviews
WHERE id = 'taken_3' AND {{
    LLMMap('Is the movie review clearly positive?', reviewText)
}} = TRUE
LIMIT 5;
