import palimpzest as pz


def run(con, pz_config: pz.QueryProcessorConfig):
    reviews = pz.MemoryDataset(
        id="reviews",
        vals=con.execute("SELECT * FROM Reviews")
        .df()
        .rename(columns={"id": "movieId"}),
    )
    reviews = reviews.filter(lambda r: r["movieId"] == "taken_3")
    reviews = reviews.sem_add_columns(
        [
            {
                "name": "positivity",
                "type": int,
                "desc": "Return 1 if the following score, as a fraction, is above 0.5, and 0 if the score is less than 0.5. Only output a single numeric value (1 or 0) with no additional commentary",
            }
        ],
        depends_on=["originalScore"],
    )
    reviews = reviews.project(["positivity"])
    reviews = reviews.average()

    return reviews.run(config=pz_config)
