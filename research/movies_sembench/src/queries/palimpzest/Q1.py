def run(con):
    reviews = pz.MemoryDataset(id="reviews", vals=self.load_data("Reviews.csv"))
    reviews = reviews.sem_filter(
        "Determine if the following movie review is clearly positive.",
        depends_on=["reviewText"],
    )
    reviews = reviews.project(["reviewId"])
    reviews = reviews.limit(5)

    reviews.run(self.palimpzest_config())
