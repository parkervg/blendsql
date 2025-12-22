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
