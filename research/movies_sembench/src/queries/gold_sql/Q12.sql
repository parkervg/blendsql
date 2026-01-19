SELECT reviewId FROM Reviews
WHERE reviewText LIKE '%Marvel%'
AND reviewText LIKE '%movie%'
AND (isTopCritic = TRUE OR creationDate LIKE '2023%')