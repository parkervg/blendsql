import palimpzest as pz


def run(con, pz_config: pz.QueryProcessorConfig):
    reviews = con.execute("SELECT * FROM Reviews").df()
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

    # Now create the dataset with pre-filtered pairs
    merged_df = pz.MemoryDataset(id="merged_df", vals=merged_df)

    # Apply the semantic condition only to the pre-filtered pairs
    merged_df = merged_df.sem_filter(
        "Is one score greater than 0.5 and the other less than 0.5?",
        depends_on=["originalScore1", "originalScore2"],
    )

    # Project to get only the columns we need
    merged_df = merged_df.project(["id1", "reviewId1", "reviewId2"])

    return merged_df.run(config=pz_config)
