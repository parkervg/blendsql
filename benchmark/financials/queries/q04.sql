SELECT DISTINCT constituents.Symbol, Action FROM constituents
LEFT JOIN account_history ON constituents.Symbol = account_history.Symbol
LEFT JOIN portfolio on constituents.Symbol = portfolio.Symbol
WHERE account_history."Run Date" > '2021-02-23'
AND ({{get_length(constituents.Name)}} > 3 OR {{test_starts_with('A', portfolio.Symbol)}})
AND portfolio.Symbol IS NOT NULL
ORDER BY LENGTH(constituents.Name) LIMIT 1