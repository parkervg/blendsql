SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)', Name
FROM account_history
LEFT JOIN constituents ON account_history.Symbol = constituents.Symbol
WHERE Sector = 'Information Technology'
AND {{test_starts_with('A', constituents.Name)}} = 1
AND lower(Action) like "%dividend%"