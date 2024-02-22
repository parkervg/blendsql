# MapIngredient
This type of ingredient applies a function on a given table/column pair to create a new column containing the function output.

For example, take the following query.

```sql 
SELECT merchant FROM transactions
    WHERE {{LLMMap('Is this a pizza shop?', 'transactions::merchant')}} = TRUE
```

`LLMMap` is one of our builtin MapIngredients. For each of the distinct values in the "merchant" column of the "transactions" table, it will create a column containing the function output.

| merchant | Is this a pizza shop? |
|----------|-----------------------|
| Domino's | 1                     |
| Safeway  | 0                     |
| Target   | 0                     |

The temporary table shown above is then combined with the original "transactions" table with an `INNER JOIN` on the "merchant" column.
