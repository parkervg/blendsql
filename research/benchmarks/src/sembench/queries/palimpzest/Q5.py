import palimpzest as pz


def run(con, pz_config: pz.QueryProcessorConfig):
    reviews_df = (
        con.execute("SELECT * FROM Reviews").df().rename(columns={"id": "movieId"})
    )
    input_df1 = reviews_df[reviews_df["movieId"] == "ant_man_and_the_wasp_quantumania"]
    input_df2 = reviews_df[reviews_df["movieId"] == "ant_man_and_the_wasp_quantumania"]
    input_df2 = input_df2.rename(
        columns={col: f"{col}_right" for col in input_df2.columns}
    )

    input1 = pz.MemoryDataset(id="input1", vals=input_df1)
    input2 = pz.MemoryDataset(id="input2", vals=input_df2)

    input3 = input1.sem_join(
        input2,
        condition="These two movie reviews express the same sentiment - either both are positive or both are negative.",
        depends_on=["reviewText", "reviewText_right"],
    )
    input3 = input3.project(["movieId", "reviewId", "reviewId_right"])
    input3 = input3.limit(10)

    return input3.run(config=pz_config)
