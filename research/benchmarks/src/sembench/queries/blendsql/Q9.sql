SELECT reviewId,
{{
    LLMMap(
        "
        Return a score from 1 to 5 representing how much the reviewer liked the movie based on the below rubric.

        Rubric:
        5: Very positive. Strong positive sentiment, indicating high satisfaction.
        4: Positive. Noticeably positive sentiment, indicating general satisfaction.
        3: Neutral. Expresses no clear positive or negative sentiment. May be factual or descriptive without emotional language.
        2: Negative. Noticeably negative sentiment, indicating some level of dissatisfaction but without strong anger or frustration.
        1: Very negative. Strong negative sentiment, indicating high dissatisfaction, frustration, or anger.
        ",
        reviewText,
        options=(1, 2, 3, 4, 5),
    )
}} AS reviewScore
FROM Reviews
WHERE id = 'ant_man_and_the_wasp_quantumania'