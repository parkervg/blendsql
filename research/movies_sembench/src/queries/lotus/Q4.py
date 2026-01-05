import pandas as pd


def run(con):
    reviews = con.execute("SELECT * FROM Reviews").df()
    taken_reviews = reviews[reviews["id"] == "taken_3"]

    if len(taken_reviews) == 0:
        return pd.DataFrame({"positivity_ratio": [0.0]})

    # Use sem_filter to select positive reviews
    positive_reviews = taken_reviews.sem_filter(
        "Determine if the following review has a positive sentiment. Review: {reviewText}."
    )

    # Compute positivity ratio
    positivity_ratio = len(positive_reviews) / len(taken_reviews)

    # Return as DataFrame
    return pd.DataFrame({"positivity_ratio": [positivity_ratio]})
