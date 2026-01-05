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
                "desc": "Return 1 if the following review is positive, and 0 if the review is not positive. Only output a single numeric value (1 or 0) with no additional commentary",
            }
        ],
        depends_on=["reviewText"],
    )
    reviews = reviews.project(["positivity"])
    reviews = reviews.average()

    output = reviews.run(config=pz_config)
