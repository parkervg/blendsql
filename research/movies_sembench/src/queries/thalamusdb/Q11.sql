SELECT r.reviewId FROM Reviews r
JOIN Movies m ON r.id = m.id
WHERE m.originalLanguage = 'Korean'
AND NLfilter(originalScore, 'The score, as a fraction, is greater than 0.5.')