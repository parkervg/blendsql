SELECT reviewId FROM Reviews
WHERE {{LLMMap('Does this review have the EXACT substring ''Marvel'' in it?', reviewText)}} = TRUE
AND {{LLMMap('Does this review have the EXACT substring ''movie'' in it?', reviewText)}} = TRUE
AND (isTopCritic = TRUE OR creationDate LIKE '2023%')
LIMIT 5