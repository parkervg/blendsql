WITH sentiment_results AS (
    SELECT
      {{
        LLMMap(
            'What is the sentiment of this review?',
            reviewText,
            options=('POSITIVE', 'NEGATIVE')
        )
      }} AS sentiment_label
      FROM Reviews
      WHERE id = 'taken_3'
) SELECT
  sentiment_label AS scoreSentiment,
  COUNT(*) AS count
FROM sentiment_results
GROUP BY sentiment_label;
