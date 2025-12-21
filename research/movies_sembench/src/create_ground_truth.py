def create_ground_truth():
    import duckdb
    import time
    import pandas as pd

    from .database_utils import iter_queries
    from .config import DUCKDB_DB_PATH

    with duckdb.connect(DUCKDB_DB_PATH, read_only=True) as con:
        print(
            f"Reviews has {con.execute('SELECT COUNT(*) FROM Reviews').fetchone()} rows"
        )
        print(
            f"Movies has {con.execute('SELECT COUNT(*) FROM Movies').fetchone()} rows"
        )

        # Run queries
        results = []
        for query_file, query_name in iter_queries("gold_sql"):
            query = open(query_file).read()
            start = time.time()
            result = con.execute(query).df()
            latency = time.time() - start
            results.append(
                {
                    "system_name": "thalamusdb",
                    "query_name": query_name,
                    "latency": latency,
                    "prediction": result.to_json(orient="split", index=False),
                }
            )
        return pd.DataFrame(results)
