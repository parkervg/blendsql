WITH scored_reviews AS (
  SELECT
    id,
    {{
        LLMMap(
            "
            Score from 1 to 5 how much did the reviewer like the movie based on provided rubrics.

            Rubrics:
            5: Very positive. Strong positive sentiment, indicating high satisfaction.
            4: Positive. Noticeably positive sentiment, indicating general satisfaction.
            3: Neutral. Expresses no clear positive or negative sentiment. May be factual or descriptive without emotional language.
            2: Negative. Noticeably negative sentiment, indicating some level of dissatisfaction but without strong anger or frustration.
            1: Very negative. Strong negative sentiment, indicating high dissatisfaction, frustration, or anger.
            ",
            reviewText,
            options=(1, 2, 3, 4, 5),
        )
    }} AS score
  FROM Reviews
) SELECT
id AS movieId,
AVG(score) AS movieScore
FROM scored_reviews
GROUP BY id;