SELECT w."Percent of Account" FROM (SELECT * FROM "portfolio" WHERE Quantity > 200 OR "Today''s Gain/Loss Percent" > 0.05) as w
JOIN geographic g ON {{
    do_join(
        w.Symbol,
        g.Symbol
    )
}} WHERE {{test_starts_with('F', w.Symbol)}}
AND w."Percent of Account" < 0.2