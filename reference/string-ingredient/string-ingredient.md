# StringIngredient
This is the simplest type of ingredient. This will output a string to be placed directly into the SQL query.

We have the `DT` function as a builtin StringIngredient.

```sql 
SELECT merchant FROM transactions
    WHERE {{DT('transactions::date', start='q2')}}
```

This will call a Python function that uses `datetime` to interpret the absolute dates which the relative phrase "q2" most likely refers to.

We do not create any new tables or perform any joins with a StringIngredient; instead, we simply get the following SQL query.

> [!NOTE]
> The below SQL interpretation of the `DT` function assumes we're calling it in December, 2022. The phrase 'q2' will be interpreted differently in, say, March 1998.

```sql 
SELECT merchant FROM transactions
    WHERE date > '2022-09-30' AND date < '2022-12-01'
```