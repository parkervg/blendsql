import pandas as pd


def run(con):
    reviews = con.execute("SELECT * FROM Reviews").df()
    # Filter for taken_3 movie
    filtered_reviews = reviews[reviews["id"] == "taken_3"]

    # Semantic map to classify sentiment
    sentiment_reviews = filtered_reviews.sem_map(
        "Classify the sentiment of this movie review as either 'POSITIVE' or 'NEGATIVE'. "
        "Only output the exact word 'POSITIVE' or 'NEGATIVE' with no additional text. "
        "Review: {reviewText}"
    )

    # Count sentiment occurrences
    # Without below, we could have 'POSITIVE\n'
    sentiment_reviews["_map"] = sentiment_reviews["_map"].apply(lambda x: x.strip("\n"))
    sentiment_counts = sentiment_reviews["_map"].value_counts().reset_index()
    sentiment_counts.columns = ["scoreSentiment", "count"]

    # Ensure we have both sentiment types in results (even if count is 0)
    expected_sentiments = ["POSITIVE", "NEGATIVE"]
    for sentiment in expected_sentiments:
        if sentiment not in sentiment_counts["scoreSentiment"].values:
            new_row = pd.DataFrame({"scoreSentiment": [sentiment], "count": [0]})
            sentiment_counts = pd.concat([sentiment_counts, new_row], ignore_index=True)

    # Sort by sentiment for consistent output
    sentiment_counts = sentiment_counts.sort_values("scoreSentiment").reset_index(
        drop=True
    )

    return sentiment_counts
