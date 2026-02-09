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
    filtered_df = reviews[reviews["id"] == "ant_man_and_the_wasp_quantumania"]

    # Self-join on id
    merged_df = filtered_df.merge(filtered_df, on="id", suffixes=("_1", "_2"))

    # Apply the condition reviewId1 < reviewId2
    merged_df = merged_df[merged_df["reviewId_1"] < merged_df["reviewId_2"]]

    # Select and rename columns
    merged_df = merged_df[
        ["originalScore_1", "originalScore_2", "id", "reviewId_1", "reviewId_2"]
    ]
    merged_df = merged_df.rename(
        columns={
            "originalScore_1": "originalScore1",
            "originalScore_2": "originalScore2",
            "id": "id1",
            "reviewId_1": "reviewId1",
            "reviewId_2": "reviewId2",
        }
    )

    # Remove duplicates (equivalent to DISTINCT)
    merged_df = merged_df.drop_duplicates()

    merged_df = merged_df.sort_values(by=["reviewId1", "reviewId2"]).reset_index(
        drop=True
    )

    joined_df = merged_df.sem_filter(
        'Is one score greater than 0.5 and the other less than 0.5? Score 1: "{originalScore1}" Score 2: "{originalScore2}"'
    )

    # Check if we got any results
    if len(joined_df) == 0:
        print("  Warning: No matching review pairs found")
        return pd.DataFrame()

    return joined_df[["id1", "reviewId1", "reviewId2"]]
