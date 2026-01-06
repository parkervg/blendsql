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
                "desc": "Return POSITIVE if the following review is positive, and NEGATIVE if the review is not positive. Only output POSITIVE or NEGATIVE with no additional commentary",
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
