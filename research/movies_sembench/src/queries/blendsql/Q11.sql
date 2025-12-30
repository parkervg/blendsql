SELECT reviewId FROM Reviews r
JOIN Movies m ON m.id = r.id
WHERE m.originalLanguage = 'Korean'
AND m.writer = 'Yeon Sang-ho'
AND {{LLMMap('Does this review have the word ''Korean'' in it?', r.reviewText)}} = TRUE
AND {{LLMMap('Is the review sentiment clearly positive?', r.reviewText)}} = TRUE