{{
    get_table_size(
        (
            WITH a AS (
                SELECT * FROM (SELECT DISTINCT * FROM portfolio) as w
                    WHERE {{test_starts_with('F', w.Symbol)}} = TRUE
            ) SELECT * FROM a WHERE LENGTH(a.Symbol) > 2
        )
    )
}}