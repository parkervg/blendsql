SELECT
{{
    LLMMap(
        'Is the score positive or negative? Return ''POSITIVE'' if the score as a fraction is greater than 0.5, and ''NEGATIVE'' otherwise.',
        originalScore,
        options=('POSITIVE', 'NEGATIVE')
    )
}} AS sentiment_label,
COUNT(*) AS count
FROM Reviews
WHERE id = 'taken_3'
AND originalScore IS NOT NULL AND originalScore LIKE '%/%' AND CAST(split_part(originalScore, '/', 1) AS FLOAT) / CAST(split_part(originalScore, '/', 2) AS FLOAT) <> 0.5
GROUP BY sentiment_label
