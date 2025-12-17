SELECT reviewId
FROM Reviews AS r
WHERE llm_filter(
    {'model_name': '<<model_name>>'},
    {'prompt': 'The following movie review is clearly positive.', 'context_columns': [{'data': r.reviewText, 'name': 'review'}]}
)
LIMIT 5;