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
    taken_reviews = reviews[reviews["id"] == "taken_3"]

    if len(taken_reviews) == 0:
        return pd.DataFrame({"positivity_ratio": [0.0]})

    # Use sem_filter to select positive reviews
    positive_reviews = taken_reviews.sem_filter(
        'Is the score, as a fraction, greater than 0.5? Score: "{originalScore}"'
    )

    # Compute positivity ratio
    positivity_ratio = len(positive_reviews) / len(taken_reviews)

    # Return as DataFrame
    return pd.DataFrame({"positivity_ratio": [positivity_ratio]})
