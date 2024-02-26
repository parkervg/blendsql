# Creating Custom BlendSQL Ingredients

All the built-in LLM ingredients inherit from the base classes `QAIngredient`, `MapIngredient`, and `JoinIngredient`.

These are intended to be helpful abstractions, so that the user can easily implement their own functions to run within a BlendSQL script.

## QAIngredient

::: blendsql.ingredients.ingredient.QAIngredient.run
    handler: python
    show_source: true

## MapIngredient

::: blendsql.ingredients.ingredient.MapIngredient.run
    handler: python
    show_source: true

## JoinIngredient

::: blendsql.ingredients.ingredient.JoinIngredient.run
    handler: python
    show_source: true