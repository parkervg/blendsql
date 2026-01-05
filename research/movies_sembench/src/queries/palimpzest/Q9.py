import palimpzest as pz


def run(con, pz_config: pz.QueryProcessorConfig):
    reviews_df = (
        con.execute("SELECT * FROM Reviews").df().rename(columns={"id": "movieId"})
    )
    filtered_reviews = reviews_df[
        reviews_df["movieId"] == "ant_man_and_the_wasp_quantumania"
    ]

    # Create dataset
    reviews = pz.MemoryDataset(id="reviews", vals=filtered_reviews)

    # Add score column using detailed rubric from Lotus
    reviews = reviews.sem_add_columns(
        [
            {
                "name": "reviewScore",
                "type": int,
                "desc": """Score from 1 to 5 how much did the reviewer like the movie based on provided rubrics.

    Rubrics:
    5: Very positive. Strong positive sentiment, indicating high satisfaction.
    4: Positive. Noticeably positive sentiment, indicating general satisfaction.
    3: Neutral. Expresses no clear positive or negative sentiment. May be factual or descriptive without emotional language.
    2: Negative. Noticeably negative sentiment, indicating some level of dissatisfaction but without strong anger or frustration.
    1: Very negative. Strong negative sentiment, indicating high dissatisfaction, frustration, or anger.

    Review: {reviewText}

    Only provide the score number (1-5) with no other comments.""",
            }
        ],
        depends_on=["reviewText"],
    )

    # Project to required columns
    reviews = reviews.project(["reviewId", "reviewScore"])

    return reviews.run(config=pz_config)
