import palimpzest as pz


def run(con, pz_config: pz.QueryProcessorConfig):
    reviews = pz.MemoryDataset(
        id="reviews",
        vals=con.execute("SELECT * FROM Reviews")
        .df()
        .rename(columns={"id": "movieId"}),
    )
    reviews = reviews.filter(lambda r: r["movieId"] == "taken_3")
    reviews = reviews.sem_filter(
        "Determine if the score, as a fraction, is greater than 0.5.",
        depends_on=["originalScore"],
    )
    reviews = reviews.project(["reviewId"])
    reviews = reviews.limit(5)
    return reviews.run(config=pz_config)
