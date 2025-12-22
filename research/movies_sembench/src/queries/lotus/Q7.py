import pandas as pd


def run(con):
    reviews = con.execute("SELECT * FROM Reviews").df()
    reviews = reviews[reviews["id"] == "ant_man_and_the_wasp_quantumania"]

    # Reset index for approximate policy
    # if hasattr(self, "policy") and self.policy == "approximate":
    #     reviews = reviews.reset_index(drop=True)

    # Semantic self-join for opposite sentiment within specific movie
    join_instruction = 'These two movie reviews express opposite sentiments - one is positive and the other is negative. Review 1: "{reviewText:left}" Review 2: "{reviewText:right}"'

    # if hasattr(self, "policy") and self.policy == "approximate":
    #     joined_df = reviews.sem_join(
    #         reviews,
    #         join_instruction=join_instruction,
    #         cascade_args=self.cascade_args,
    #     )
    # else:
    joined_df = reviews.sem_join(reviews, join_instruction=join_instruction)

    # Filter out self-matches (same reviewId)
    joined_df = joined_df[joined_df["reviewId:left"] != joined_df["reviewId:right"]]

    # Check if we got any results
    if len(joined_df) == 0:
        print("  Warning: No matching opposite sentiment review pairs found")
        return pd.DataFrame()

    # Select and rename relevant columns to match expected format (evaluator needs id, reviewId, reviewId2)
    result_df = joined_df[["id:left", "reviewId:left", "reviewId:right"]].copy()
    result_df.columns = ["id", "reviewId", "reviewId2"]

    return result_df
