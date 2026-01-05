SELECT reviewId FROM Reviews
WHERE {{LLMMap('Does this review have the EXACT substring ''Marvel'' in it?', reviewText)}} = TRUE
AND {{LLMMap('Does this review have the EXACT substring ''action'' in it?', reviewText)}} = TRUE
AND (creationDate LIKE '2019%' OR creationDate LIKE '2023%')
LIMIT 2