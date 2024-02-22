# JoinIngredient
Handles the logic of semantic `JOIN` clauses between tables.

For example:
```sql
SELECT Capitals.name, State.name FROM Capitals
    JOIN {{
        LLMJoin(
            'Align state to capital', 
            left_on='States::name', 
            right_on='Capitals::name'
        )
    }}
```
The above example hints at a database schema that would make [E.F Codd](https://en.wikipedia.org/wiki/Edgar_F._Codd) very angry: why do we have two separate tables `States` and `Capitals` with no foreign key to join the two?

BlendSQL was built to interact with tables "in-the-wild", and many (such as those on Wikipedia) do not have these convenient properties of well-designed relational models.

For this reason, we can leverage the internal knowledge of a pre-trained LLM to do the `JOIN` operation for us.
