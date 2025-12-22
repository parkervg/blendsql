import pandas as pd


def run(con):
    reviews = con.execute("SELECT * FROM Reviews").df()
    reviews = reviews[reviews["id"] == "ant_man_and_the_wasp_quantumania"]

    # Reset index for approximate policy
    # if hasattr(self, "policy") and self.policy == "approximate":
    #     reviews = reviews.reset_index(drop=True)

    # Check if we have reviews for this movie
    if len(reviews) == 0:
        print(
            "  Warning: No reviews found for movie 'ant_man_and_the_wasp_quantumania'"
        )
        return pd.DataFrame()

    # Semantic self-join for same sentiment within specific movie
    join_instruction = 'These two movie reviews express the same sentiment - either both are positive or both are negative. Review 1: "{reviewText:left}" Review 2: "{reviewText:right}"'

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
        print("  Warning: No matching review pairs found")
        return pd.DataFrame()

    # Limit to 10 results
    final_result = joined_df.head(10)

    # Select and rename relevant columns to match expected format (evaluator needs id, reviewId, reviewId2)
    result_df = final_result[["id:left", "reviewId:left", "reviewId:right"]].copy()
    result_df.columns = ["id", "reviewId", "reviewId2"]

    return result_df
