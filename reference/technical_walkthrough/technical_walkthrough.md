All the below logic can be found in the `blend()` function from [`blendsql/blendsql.py`](execute-blendsql.md).


## Example 

We can take the following query as an example.

```sql
--- 'Show me dividends from tech companies that manufacture cell phones'
SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)', Name
    FROM account_history
    LEFT JOIN constituents ON account_history.Symbol = constituents.Symbol
    WHERE constituents.Sector = 'Information Technology'
    AND {{
           LLM(
               'does this company manufacture cell phones?',
               'constituents::Name',
            )
    }} = 1
    AND lower(account_history.Action) like "%dividend%"
```
#### 1) Generate a Session UUID
This uuid allows us to create new temporary tables containing the output of our BlendSQL functions rather than overwriting the original, underlying SQL tables.

```python
import uuid 

session_uuid = str(uuid.uuid4())[:5]
```

#### 2) Identify All Subqueries
Here, we define subqueries as any select statement from a single table. These may or may not have their own `SELECT` clause.

In our example, we only have a single query.

#### 3) For Each Table, Generate Select Statements

Iterating through our subqueries, we now need to write each table reference as its own `SELECT` statement.

To do this, we iterate through each table and get the following queries.

Notice how in the `account_history` query, we change the specific columns to the `*`, so we get everything.

```sql
SELECT * FROM account_history
    WHERE lower(account_history.Action) like "%dividend%"
```

```sql
SELECT * FROM constituents
    WHERE constituents.Sector = 'Information Technology'
    AND {{
           LLM(
               'does this company manufacture cell phones?',
               'constituents::Name',
            )
    }} = 1
```

#### 4) Abstract Away Selects

In our `constituents` subquery, we have an expensive LLM operation.

In order to make sure we pass minimal required data to our BlendSQL ingredients while still honoring the SQL logic, we abstract away external functions to `True` to calculate the theoretical upper bound of data that might get returned.

```sql
-- Abstracted query
SELECT * FROM constituents
    WHERE constituents.Sector = 'Information Technology'
    AND TRUE
```

We execute each of these queries and assign them to new temporary tables, `f"{session_uuid}_{tablename}_{subquery_idx}"`.

#### 4) Execute BlendSQL Ingredients on our New Tables
Now we can execute some external functions.

For example, if we have a session_id of '1234':
```sql
SELECT Symbol FROM "1234_constituents_0" 
    WHERE sector = 'Information Technology' 
    AND {{
            LLM(
                'does this company manufacture cell phones?', 
                'constituents::Name'
            )
        }} = 1
```

The table "1234_constituents_0" now only has those entries where `sector = 'Information Technology'`. This minimizes the data that the `{{LLM()}}` call actually needs to process.

Once we've executed our functions, we now have a table with a new column, `'does this company manufacture cell phones?'`.

We do a left join with the original `constituents` table to create the new session table "1234_constituents". 

Then, we can move to the next subquery. In this case, there is no BlendSQL ingredient, so we're done with our processing.

#### 5) Execute our Final SQL Query
At the end of our processing, we have the underlying SQL tables and query in a state that we can execute it like any other SQLite script. 

```sql
SELECT "Run Date", Account, Action, ROUND("Amount ($)", 2) AS 'Total Dividend Payout ($$)', Name
    FROM account_history
    LEFT JOIN '1c0b_constituents' ON account_history.Symbol = '1c0b_constituents'.Symbol
    WHERE '1c0b_constituents'.Sector = 'Information Technology'
    AND {{
           LLM(
               'does this company manufacture cell phones?',
               'constituents::Name',
            )
    }} = 1
    AND lower(account_history.Action) like "%dividend%"
```

Notice how `account_history` was not modified by a session_id, but `constituents` was. This is because the logic of filtering by `LIKE %dividend%` can just be done using raw SQL on the original table, we don't need any complicated BlendSQL processing.

