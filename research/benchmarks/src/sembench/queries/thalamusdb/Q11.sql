SELECT r.reviewId FROM Reviews r
JOIN Movies m ON r.id = m.id
WHERE m.originalLanguage = 'Korean'
AND NLfilter(r.reviewText, 'The movie review has a positive sentiment')