import pandas as pd
from textwrap import dedent


def run(con):
    reviews = con.execute("SELECT * FROM Reviews").df()

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

    # Use sem_map for scoring all reviews - simpler and more reliable
    scored_reviews = reviews.sem_map(scoring_prompt)

    # Extract scores from the _map column and group by movie to calculate averages
    movie_scores = []
    grouped_scored = scored_reviews.groupby("id")

    for movie_id, movie_scored_reviews in grouped_scored:
        valid_scores = []
        for _, row in movie_scored_reviews.iterrows():
            score = row["_map"]
            try:
                numeric_score = float(score)
                if 1 <= numeric_score <= 5:
                    valid_scores.append(numeric_score)
                else:
                    valid_scores.append(3.0)  # Default to neutral
            except (ValueError, TypeError):
                valid_scores.append(3.0)  # Default to neutral

        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)
        else:
            avg_score = 3.0  # Default to neutral

        movie_scores.append({"movieId": movie_id, "movieScore": round(avg_score, 2)})

    result_df = pd.DataFrame(movie_scores)
    return result_df
