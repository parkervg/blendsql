SELECT reviewId FROM Reviews
WHERE NLfilter(reviewText, 'This review has the EXACT substring ''Marvel'' in it')
AND NLfilter(reviewText, 'This review has the EXACT substring ''movie'' in it')
AND (creationDate LIKE '2019%' OR creationDate LIKE '2023%')
LIMIT 2