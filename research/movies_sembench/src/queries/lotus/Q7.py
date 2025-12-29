import pandas as pd


def run(con):
    reviews = con.execute("SELECT * FROM Reviews").df()
    filtered_df = reviews[reviews["id"] == "ant_man_and_the_wasp_quantumania"]

    # Self-join on id
    merged_df = filtered_df.merge(filtered_df, on="id", suffixes=("_1", "_2"))

    # Apply the condition reviewId1 < reviewId2
    merged_df = merged_df[merged_df["reviewId_1"] < merged_df["reviewId_2"]]

    # Select and rename columns
    merged_df = merged_df[
        ["reviewText_1", "reviewText_2", "id", "reviewId_1", "reviewId_2"]
    ]
    merged_df = merged_df.rename(
        columns={
            "reviewText_1": "reviewText1",
            "reviewText_2": "reviewText2",
            "id": "id1",
            "reviewId_1": "reviewId1",
            "reviewId_2": "reviewId2",
        }
    )

    # Remove duplicates (equivalent to DISTINCT)
    merged_df = merged_df.drop_duplicates()

    joined_df = merged_df.sem_filter(
        'These two movie reviews express opposite sentiment - either both are positive or both are negative. Review 1: "{reviewText1}" Review 2: "{reviewText2}"'
    )

    # Check if we got any results
    if len(joined_df) == 0:
        print("  Warning: No matching review pairs found")
        return pd.DataFrame()

    return joined_df[["id1", "reviewId1", "reviewId2"]]
