SELECT reviewId FROM Reviews
WHERE NLfilter(reviewText, 'This review has the EXACT substring ''Marvel'' in it')
AND NLfilter(reviewText, 'This review has the EXACT substring ''movie'' in it')
AND (isTopCritic = TRUE OR creationDate LIKE '2023%')
LIMIT 5