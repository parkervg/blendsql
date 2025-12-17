SELECT
  sentiment_label AS scoreSentiment,
  COUNT(*) AS count
FROM (
  SELECT
  {{
    LLMMap(
        'What is the sentiment of this review?',
        r.reviewText,
        options=('POSITIVE', 'NEGATIVE')
    )
  }} AS sentiment_label
  FROM Reviews AS r
  WHERE r.id = 'taken_3'
) AS sentiment_results
GROUP BY sentiment_label;