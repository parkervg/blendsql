SELECT reviewId
FROM Reviews
WHERE id = 'taken_3' AND {{
    LLMMap(
        'Is the score, as a fraction, greater than 0.5?',
        originalScore
    )
}} = TRUE
AND originalScore IS NOT NULL AND originalScore LIKE '%/%' AND CAST(split_part(originalScore, '/', 1) AS FLOAT) / CAST(split_part(originalScore, '/', 2) AS FLOAT) <> 0.5
LIMIT 5
