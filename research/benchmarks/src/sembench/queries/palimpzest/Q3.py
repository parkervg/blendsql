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
        "Determine if the following movie review is clearly positive.",
        depends_on=["reviewText"],
    )
    reviews = reviews.count()

    return reviews.run(config=pz_config)
