SELECT COUNT(*) AS positive_review_counts
FROM Reviews AS r
WHERE r.id == 'taken_3' AND llm_filter(
      {'model_name': '<<model_name>>'}, 
      {'prompt': 'The following movie review is clearly positive.', 'context_columns': [{'data': r.reviewText, 'name': 'review'}]},
  );