def compute_recall(gt_ids, ids, cutoff=1000):
    return len(set(gt_ids).intersection(set(ids[:cutoff]))) / len(gt_ids)


def compute_precision(gt_ids, ids, cutoff=1000):
    if len(ids[:cutoff]) == 0:
        return 0
    else:
        return len(set(gt_ids).intersection(set(ids[:cutoff]))) / len(ids[:cutoff])


def compute_rank_precision(gt_ids, ids, cutoff=1000):
    gt_ids = [
        gt_id.strip().lower().replace("'", "").replace("^", "") for gt_id in gt_ids
    ]
    ids = [id.strip().lower().replace("'", "").replace("^", "") for id in ids]
    if len(ids[:cutoff]) == 0:
        return 0
    else:
        divisor = min(len(gt_ids), cutoff)
        count = 0
        for i in range(min(cutoff, len(ids))):
            if ids[i] in gt_ids:
                count += 1
        return count / divisor


def compute_metrics(
    res_df, gt_col_name="reactions_list", pred_col_name="pred_reaction"
) -> pd.DataFrame:
    res_df["rank_precision@5"] = res_df.apply(
        lambda x: compute_rank_precision(x[gt_col_name], x[pred_col_name], cutoff=5),
        axis=1,
    )
    res_df["rank-precision@10"] = res_df.apply(
        lambda x: compute_rank_precision(x[gt_col_name], x[pred_col_name], cutoff=10),
        axis=1,
    )

    res_df["rank-precision@25"] = res_df.apply(
        lambda x: compute_rank_precision(x[gt_col_name], x[pred_col_name], cutoff=25),
        axis=1,
    )

    res_df["num_ids"] = res_df.apply(lambda x: len(x[pred_col_name]), axis=1)

    # take subset of df with metrics
    df = res_df[
        [
            col
            for col in res_df.columns
            if "@" in col or "latency" in col or "num_ids" in col
        ]
    ]

    return df
