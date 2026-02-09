WITH self_joined_reviews AS (
    SELECT DISTINCT
    r1.originalScore AS originalScore1,
    r2.originalScore AS originalScore2,
    r1.id as id1,
    r1.reviewId as reviewId1,
    r2.reviewId as reviewId2
    FROM Reviews r1
    JOIN Reviews r2
    ON r1.id = r2.id
    AND r1.reviewId < r2.reviewId
    WHERE r1.id = 'ant_man_and_the_wasp_quantumania'
    AND r2.id = 'ant_man_and_the_wasp_quantumania'
    AND originalScore1 IS NOT NULL AND originalScore1 LIKE '%/%' AND CAST(split_part(originalScore1, '/', 1) AS FLOAT) / CAST(split_part(originalScore1, '/', 2) AS FLOAT) <> 0.5
    AND originalScore2 IS NOT NULL AND originalScore2 LIKE '%/%' AND CAST(split_part(originalScore2, '/', 1) AS FLOAT) / CAST(split_part(originalScore2, '/', 2) AS FLOAT) <> 0.5
    ORDER BY r1.reviewId, r2.reviewId
) SELECT id1, reviewId1, reviewId2 FROM self_joined_reviews
WHERE {{
    LLMMap(
        'Are both scores greater than 0.5?',
        originalScore1,
        originalScore2
    )
}} = TRUE
LIMIT 10