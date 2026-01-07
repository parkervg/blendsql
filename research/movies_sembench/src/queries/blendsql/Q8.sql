SELECT
{{
    LLMMap(
        'What is the sentiment of this review? Classify as either ''POSITIVE'' or ''NEGATIVE''.',
        reviewText,
        options=('POSITIVE', 'NEGATIVE')
    )
}} AS sentiment_label,
COUNT(*) AS count
FROM Reviews
WHERE id = 'taken_3'
GROUP BY sentiment_label
