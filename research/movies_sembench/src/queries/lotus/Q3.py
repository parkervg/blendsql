import pandas as pd


def run(con):
    reviews = con.execute("SELECT * FROM Reviews").df()
    filtered_reviews = reviews[reviews["id"] == "taken_3"]

    # Semantic filter for positive reviews
    positive_reviews = filtered_reviews.sem_filter(
        "Determine if the following review is clearly positive. Review: {reviewText}"
    )

    # Get count
    positive_review_cnt = positive_reviews.shape[0]

    # Return as DataFrame
    result_df = pd.DataFrame({"positive_review_cnt": [positive_review_cnt]})

    return result_df
