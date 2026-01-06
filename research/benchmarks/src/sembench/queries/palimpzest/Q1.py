import palimpzest as pz


def run(con, pz_config: pz.QueryProcessorConfig):
    reviews = pz.MemoryDataset(
        id="reviews", vals=con.execute("SELECT * FROM Reviews").df()
    )
    reviews = reviews.sem_filter(
        "Determine if the following movie review is clearly positive.",
        depends_on=["reviewText"],
    )
    reviews = reviews.project(["reviewId"])
    reviews = reviews.limit(5)

    return reviews.run(config=pz_config)
