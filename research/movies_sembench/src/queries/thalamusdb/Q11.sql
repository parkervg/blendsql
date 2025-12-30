SELECT reviewId FROM Reviews r
JOIN Movies m ON m.id = r.id
WHERE m.originalLanguage = 'Korean'
AND m.writer = 'Yeon Sang-ho'
AND NLfilter(r.reviewText, 'this review has the word ''Korean'' in it')
AND NLfilter(r.reviewText, 'the review sentiment is clearly positive')