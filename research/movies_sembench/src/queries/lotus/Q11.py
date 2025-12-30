import pandas as pd


def run(con):
    reviews = con.execute("SELECT * FROM Reviews").df()
    movies = con.execute("SELECT * FROM Movies").df()

    # First, apply join
    merged_df = pd.merge(movies, reviews, left_on="id", right_on="id")

    # Then apply filter
    merged_df = merged_df[
        (merged_df["originalLanguage"] == "Korean")
        & (merged_df["writer"] == "Yeon Sang-ho")
    ]

    # Semantic filter for clearly positive reviews
    filtered_reviews = merged_df.sem_filter(
        'Determine if the following movie review is clearly positive. Review: "{reviewText}".'
    )

    filtered_reviews = filtered_reviews.sem_filter(
        "Determine if the following movie review has the word "
        "Korean"
        ' in it. Review: "{reviewText}".'
    )

    # Check if we got any results
    if len(filtered_reviews) == 0:
        print("  Warning: No reviews found for Q11")
        return pd.DataFrame()

    return filtered_reviews[["reviewId"]]
