SELECT 
  sentiment_label AS scoreSentiment,
  COUNT(*) AS count
FROM (
  SELECT 
    CASE WHEN llm_filter(
      {'model_name': '<<model_name>>'}, 
      {
        'prompt': 'The following movie review is clearly positive.',
        'context_columns': [{'data': r.reviewText, 'name': 'review'}]
      }
     ) THEN 'POSITIVE' ELSE 'NEGATIVE'
    END AS sentiment_label
  FROM Reviews AS r
  WHERE r.id = 'taken_3'
) AS sentiment_results
GROUP BY sentiment_label;