SELECT reviewId
FROM Reviews
WHERE id = 'taken_3' AND {{
LLMMap(
    'What is the sentiment of this review?',
    reviewText,
    options=('POSITIVE', 'NEGATIVE')
)
}} = 'POSITIVE'
LIMIT 5;
