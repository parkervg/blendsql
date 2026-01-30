SELECT scoreSentiment, COUNT(*) AS count FROM Reviews WHERE id = 'taken_3'
AND originalScore IS NOT NULL AND originalScore LIKE '%/%' AND CAST(split_part(originalScore, '/', 1) AS FLOAT) / CAST(split_part(originalScore, '/', 2) AS FLOAT) <> 0.5
GROUP BY scoreSentiment;