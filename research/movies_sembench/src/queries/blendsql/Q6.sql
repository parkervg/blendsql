WITH self_joined_reviews AS (
    SELECT DISTINCT
    r1.reviewText AS reviewText1,
    r2.reviewText AS reviewText2,
    r1.id as id1,
    r1.reviewId as reviewId1,
    r2.reviewId as reviewId2
    FROM Reviews r1
    JOIN Reviews r2
    ON r1.id = r2.id
    AND r1.reviewId < r2.reviewId
    WHERE r1.id = 'ant_man_and_the_wasp_quantumania'
    AND r2.id = 'ant_man_and_the_wasp_quantumania'
    ORDER BY r1.reviewId, r2.reviewId
) SELECT id1, reviewId1, reviewId2 FROM self_joined_reviews
WHERE {{
    LLMMap(
        'Do the two movie reviews express opposite sentiments? I.e, one is positive and one is negative.',
        reviewText1,
        reviewText2
    )
}} = TRUE
LIMIT 10