SELECT DISTINCT
r1.id, r1.reviewId as reviewId1, r2.reviewId as reviewId2
FROM Reviews r1
JOIN Reviews r2
ON r1.id = r2.id
AND r1.reviewId < r2.reviewId
WHERE r1.id = 'ant_man_and_the_wasp_quantumania'
AND r2.id = 'ant_man_and_the_wasp_quantumania'
AND r1.originalScore IS NOT NULL AND r1.originalScore LIKE '%/%' AND CAST(split_part(r1.originalScore, '/', 1) AS FLOAT) / CAST(split_part(r1.originalScore, '/', 2) AS FLOAT) <> 0.5
AND r2.originalScore IS NOT NULL AND r2.originalScore LIKE '%/%' AND CAST(split_part(r2.originalScore, '/', 1) AS FLOAT) / CAST(split_part(r2.originalScore, '/', 2) AS FLOAT) <> 0.5
and NLjoin(r1.originalScore, r2.originalScore, 'One score is greater than 0.5, and the other less than 0.5.')
limit 10