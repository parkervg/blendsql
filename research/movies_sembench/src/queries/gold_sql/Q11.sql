SELECT reviewId FROM Reviews r
JOIN Movies m ON m.id = r.id
WHERE m.originalLanguage = 'Korean'
AND m.writer = 'Yeon Sang-ho'
AND r.reviewText LIKE '% Korean %'
AND r.scoreSentiment = 'POSITIVE'