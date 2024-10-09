---
hide:
  - toc
---

# Some Cool Things by Example

### Inferring Regular Expression Constraints
```sql
SELECT "Name" FROM parks
WHERE "Location" = 'Alaska'
ORDER BY {{
    LLMMap(
        question='Size in km2?',
        context='parks::Area'
    )
}} DESC LIMIT 1
```
By virtue of the `ORDER BY` clause, we assume that the output of the `{{LLMMap}` ingredient should be a numeric. BlendSQL constrains the generation of the language model, then, to the regular expression corresponding to a integer (or floating point) `(((\d|\.)+|-);){n}`, where `n` is the number of values in the `Area` column, and `-` represents a null value.

### Automatic `options` Injection
```sql
SELECT "Location", "Name" AS "Park Protecting Ash Flow" FROM parks
    WHERE "Name" = {{
      LLMQA(
        'Which park protects an ash flow?',
        (SELECT "Name", "Description" FROM parks)
      )
  }}
```
We can omit the `options` argument, and BlendSQL will automatically infer the `options="parks::Name"` argument.

### Referencing CTE, Passing in Enumerated Options
```sql
WITH w AS (
    SELECT *
    FROM account_history
    WHERE Symbol IS NOT NULL
) SELECT Symbol, {{
    LLMMap(
        'Sells cell phones?',
        'w::Description',
        options='t;f'
    )
}} FROM w
```
The `context` arg can reference a table created from a CTE, and our `options` value can be a semi-colon seperated list of strings.

### Conditional Materializing of CTE Statements

```sql
WITH a AS (
    SELECT * FROM portfolio WHERE Quantity > 200
), b AS
(
    SELECT Symbol FROM portfolio AS w WHERE w.Symbol LIKE "A%"
),
SELECT * FROM a WHERE {{starts_with('F', 'a::Symbol')}} = TRUE
JOIN b ON a.Symbol = b.Symbol
```
We only eagerly materialize a table from a CTE if it's used within an ingredient. Above, BlendSQL will materialize the `a` table, but not `b`.
