---
hide:
  - toc
---
# LLMMap
![ingredients](../../img/
/LLMMap.jpg)

## Usage 
### `LLMMap`
::: blendsql.ingredients.builtin.map.main.LLMMap
    handler: python
    options:
      show_source: false
      show_root_heading: false    
      members:
      - from_args
      - run

## Description
This type of ingredient applies a function on a given column to create a new column containing the function's output.

In more formal terms, it is a unary scalar function, much like [`LENGTH`](https://www.sqlite.org/lang_corefunc.html#length) or [`ABS`](https://www.sqlite.org/lang_corefunc.html#abs) in standard SQLite.

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

