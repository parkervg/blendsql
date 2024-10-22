---
hide:
  - toc
---
# QAIngredient
![ingredients](../../img/
/LLMQA.jpg)

## Usage 
### `LLMQA`
::: blendsql.ingredients.builtin.qa.main.LLMQA
    handler: python
    options:
      show_source: false
      show_root_heading: false    
      members:
      - from_args
      - run

## Description
Sometimes, simply selecting data from a given database is not enough to sufficiently answer a user's question.

The `QAIngredient` is designed to return data of variable types, and is best used in cases when we either need:
1) Unstructured, free-text responses ("Give me a summary of all my spending in coffe")
2) Complex, unintuitive relationships extracted from table subsets ("How many consecutive days did I spend in coffee?")
3) Multi-hop reasoning from unstructured data, grounded in a structured schema (using the `options` arg)

Formally, this is an [aggregate function](https://www.sqlite.org/lang_aggfunc.html) which transforms a table subset into a single value. 

The following query demonstrates usage of the builtin `LLMQA` ingredient.

```sql
{{
    LLMQA(
        'How many consecutive days did I buy stocks in Financials?', 
        (
            SELECT account_history."Run Date", account_history.Symbol, constituents."Sector"
              FROM account_history
              LEFT JOIN constituents ON account_history.Symbol = constituents.Symbol
              WHERE Sector = "Financials"
              ORDER BY "Run Date" LIMIT 5
        )
    )
}} 
```
This is slightly more complicated than the rest of the ingredients. 

Behind the scenes, we wrap the call to `LLMQA` in a `SELECT` clause, ensuring that the ingredient's output gets returned.
```sql 
SELECT {{QAIngredient}}
```
The LLM gets both the question asked, alongside the subset of the SQL database fetched by our subquery.

| **"Run Date"** | **Symbol** | **Sector** |
|----------------|------------|------------|
| 2022-01-14     | HBAN       | Financials |
| 2022-01-20     | AIG        | Financials |
| 2022-01-24     | AIG        | Financials |
| 2022-01-24     | NTRS       | Financials |
| 2022-01-25     | HBAN       | Financials |


From examining this table, we see that we bought stocks in the Financials sector 2 consecutive days (2022-01-24, and 2022-01-25).
The LLM answers the question in an end-to-end manner, returning the result `2`.

The `QAIngredient` can be used as a standalone end-to-end QA tool, or as a component within a larger BlendSQL query.

For example, the BlendSQL query below translates to the valid (but rather confusing) question: 

"Show me stocks in my portfolio, whose price is greater than the number of consecutive days I bought Financial stocks multiplied by 10. Only display those companies which offer a media streaming service."
```sql
 SELECT Symbol, "Last Price" FROM portfolio WHERE "Last Price" > {{
  LLMQA(
        'How many consecutive days did I buy stocks in Financials?', 
        (
            SELECT account_history."Run Date", account_history.Symbol, constituents."Sector"
              FROM account_history
              LEFT JOIN constituents ON account_history.Symbol = constituents.Symbol
              WHERE Sector = "Financials"
              ORDER BY "Run Date" LIMIT 5
        )
    )
  }} * 10
  AND {{LLMMap('Offers a media streaming service?', 'portfolio::Description')}} = 1
```
## Constrained Decoding with `options`
Perhaps we want the answer to the above question in a different format. We call our LLM ingredient in a constrained setting by passing a `options` argument, where we provide either semicolon-separated options, or a reference to a column.

```sql
{{
    LLMQA(
        'How many consecutive days did I buy stocks in Financials?', 
        (
            SELECT account_history."Run Date", account_history.Symbol, constituents."Sector"
              FROM account_history
              LEFT JOIN constituents ON account_history.Symbol = constituents.Symbol
              WHERE Sector = "Financials"
              ORDER BY "Run Date" LIMIT 5
        )
        options='one consecutive day!;two consecutive days!;three consecutive days!'
    )
}}
```

Running the above BlendSQL query, we get the output `two consecutive days!`.

This `options` argument can also be a reference to a given column.

For example (from the [HybridQA dataset](https://hybridqa.github.io/)): 

```sql 
 SELECT capacity FROM w WHERE venue = {{
        LLMQA(
            'Which venue is named in honor of Juan Antonio Samaranch?',
            (SELECT title, content FROM documents WHERE content MATCH 'venue'),
            options='w::venue'
        )
}}
```

Or, from our running example:
```sql
{{
  LLMQA(
      'Which did i buy the most?',
      (
        SELECT account_history."Run Date", account_history.Symbol, constituents."Sector"
          FROM account_history
          LEFT JOIN constituents ON account_history.Symbol = constituents.Symbol
          WHERE Sector = "Financials"
          ORDER BY "Run Date" LIMIT 5
      )
      options='account_history::Symbol'
  )
}}
```

The above BlendSQL will yield the result `AIG`, since it appears in the `Symbol` column from `account_history`.