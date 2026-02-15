import palimpzest as pz
from palimpzest.core.elements.groupbysig import GroupBySig


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
                "name": "sentiment",
                "type": str,
                "desc": "Classify the sentiment of this movie review as either 'POSITIVE' or 'NEGATIVE'. "
                "Return 'POSITIVE' if the score as a fraction is greater than 0.5, and 'NEGATIVE' otherwise."
                "Only output the exact word 'POSITIVE' or 'NEGATIVE' with no additional text. ",
            }
        ],
        depends_on=["reviewText"],
    )
    reviews = reviews.project(["sentiment"])
    gby_desc = GroupBySig(
        group_by_fields=["sentiment"],
        agg_funcs=["count"],
        agg_fields=["sentiment"],
    )
    reviews = reviews.groupby(gby_desc)

    return reviews.run(config=pz_config)
