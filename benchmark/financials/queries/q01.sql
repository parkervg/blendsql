SELECT Symbol, "North America", "Japan" FROM geographic
WHERE geographic.Symbol IN (
    SELECT Symbol FROM portfolio
    WHERE {{starts_with('A', 'portfolio::Description')}} = 1
    AND portfolio.Symbol in (
        SELECT Symbol FROM constituents
        WHERE constituents.Sector = 'Information Technology'
    )
)