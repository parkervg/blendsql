SELECT distinct r1.id as id, r1.reviewId as reviewId1, r2.reviewId as reviewId2
FROM Reviews  AS r1
JOIN Reviews  AS r2
ON r1.id = r2.id AND r1.reviewId < r2.reviewId
WHERE r1.id = 'ant_man_and_the_wasp_quantumania' AND llm_filter(
  {'model_name': '<<model_name>>'},
  {
    'prompt': 'These two movie reviews express the same sentiment - either both are positive or both are negative',
    'context_columns': [
        {'data': r1.reviewText, 'name': 'review1'},
        {'data': r2.reviewText, 'name': 'review2'}
    ]
  }
)
LIMIT 10;
