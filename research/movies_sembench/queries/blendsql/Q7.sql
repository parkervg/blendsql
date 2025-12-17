SELECT r1.id as id, r1.reviewId as reviewId1, r2.reviewId as reviewId2
FROM Reviews  AS r1
JOIN Reviews  AS r2
ON r1.id = r2.id AND r1.reviewId <> r2.reviewId
WHERE r1.id = 'ant_man_and_the_wasp_quantumania'
AND {{LLMMap('Do {} and {} express opposite sentiments? One is positive, and one is negative.', r1.reviewText, r2.reviewText)}} = TRUE
