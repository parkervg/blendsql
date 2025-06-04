SELECT Symbol, "North America", "Japan" FROM geographic
WHERE geographic.Symbol IN (
    SELECT Symbol FROM portfolio
    WHERE {{test_starts_with('A', Description)}} = 1
    AND Symbol in (
        SELECT Symbol FROM constituents
        WHERE constituents.Sector = 'Information Technology'
    )
)