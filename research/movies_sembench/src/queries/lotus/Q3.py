import pandas as pd


def run(con):
    reviews = con.execute("SELECT * FROM Reviews").df()

    def is_valid_fraction_not_half(score):
        if pd.isna(score) or "/" not in str(score):
            return False
        try:
            parts = str(score).split("/")
            numerator = float(parts[0])
            denominator = float(parts[1])
            return denominator != 0 and (numerator / denominator) != 0.5
        except (ValueError, IndexError):
            return False

    mask = reviews["originalScore"].apply(is_valid_fraction_not_half)
    reviews = reviews[mask]
    filtered_reviews = reviews[reviews["id"] == "taken_3"]

    # Semantic filter for positive reviews
    positive_reviews = filtered_reviews.sem_filter(
        'Is the score, as a fraction, greater than 0.5? Score: "{originalScore}"'
    )

    # Get count
    positive_review_cnt = positive_reviews.shape[0]

    # Return as DataFrame
    result_df = pd.DataFrame({"positive_review_cnt": [positive_review_cnt]})

    return result_df
