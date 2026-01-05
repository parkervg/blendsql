import pandas as pd
from textwrap import dedent
import random

random.seed(42)

DEFAULT_VALUE = -100


def run(con):
    reviews = con.execute("SELECT * FROM Reviews").df()
    # Filter for ant_man_and_the_wasp_quantumania movie
    filtered_reviews = reviews[reviews["id"] == "ant_man_and_the_wasp_quantumania"]

    if len(filtered_reviews) == 0:
        print(
            "  Warning: No reviews found for movie 'ant_man_and_the_wasp_quantumania'"
        )
        return pd.DataFrame(columns=["reviewId", "reviewScore"])

    # Define the scoring prompt based on BigQuery implementation
    scoring_prompt = dedent(
        """Return a score from 1 to 5 representing how much the reviewer liked the movie based on the below rubric.

        Rubric:
        5: Very positive. Strong positive sentiment, indicating high satisfaction.
        4: Positive. Noticeably positive sentiment, indicating general satisfaction.
        3: Neutral. Expresses no clear positive or negative sentiment. May be factual or descriptive without emotional language.
        2: Negative. Noticeably negative sentiment, indicating some level of dissatisfaction but without strong anger or frustration.
        1: Very negative. Strong negative sentiment, indicating high dissatisfaction, frustration, or anger.
        
        Review: {reviewText}

        Only provide the score number (1-5) with no other comments."""
    )

    # Use sem_map for scoring - returns scores directly
    scored_reviews = filtered_reviews.sem_map(scoring_prompt)

    # Extract scores from the _map column
    results = []
    for _, row in scored_reviews.iterrows():
        score = row["_map"]
        # Ensure score is numeric and within range 1-5
        try:
            numeric_score = float(score)
            if 1 <= numeric_score <= 5:
                results.append(
                    {
                        "reviewId": row["reviewId"],
                        "reviewScore": numeric_score,
                    }
                )
            else:
                results.append(
                    {"reviewId": row["reviewId"], "reviewScore": DEFAULT_VALUE}
                )
        except (ValueError, TypeError):
            results.append({"reviewId": row["reviewId"], "reviewScore": DEFAULT_VALUE})
    result_df = pd.DataFrame(results)
    return result_df
