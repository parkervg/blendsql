---
hide:
  - toc
---
# LLMJoin
![ingredients](../../img/
/LLMJoin.jpg)

## Usage
### `LLMJoin`
::: blendsql.ingredients.builtin.join.main.LLMJoin
    handler: python
    options:
      show_source: false
      show_root_heading: false
      members:
      - from_args

## Description
This ingredient handles the logic of semantic `JOIN` clauses between tables.

In other words, it creates a custom mapping between a pair of value sets. Behind the scenes, this mapping is then used to create an auxiliary table to use in carrying out an [`INNER JOIN`](https://www.sqlite.org/optoverview.html#joins).

For example:
```sql
SELECT Capitals.name, State.name FROM Capitals
    JOIN {{
        LLMJoin(
            'Capitals::name',
            'States::name',
            question='Align state to capital.',
        )
    }}
```
The above example hints at a database schema that would make [E.F Codd](https://en.wikipedia.org/wiki/Edgar_F._Codd) very angry: why do we have two separate tables `States` and `Capitals` with no foreign key to join the two?

BlendSQL was built to interact with tables "in-the-wild", and many (such as those on Wikipedia) do not have these convenient properties of well-designed relational models.

For this reason, we can leverage the internal knowledge of a pre-trained LLM to do the `JOIN` operation for us.
