SELECT AVG(
    CASE WHEN
    llm_filter(
        {'model_name': '<<model_name>>'}, 
        {'prompt': 'Is it a positive review?', 'context_columns': [{'data': r.reviewText, 'name': 'review'}]}
    )
    THEN 1 ELSE 0
    END
) AS positive_reviews_ratio
FROM Reviews AS r
WHERE id = 'taken_3';
