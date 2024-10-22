---
hide:
  - toc
---
# Creating Custom BlendSQL Ingredients

All the built-in LLM ingredients inherit from the base classes `QAIngredient`, `MapIngredient`, `JoinIngredient`, and `AliasIngredient`.

These are intended to be helpful abstractions, so that the user can easily implement their own functions to run within a BlendSQL script.

The processing logic for a custom ingredient should go in a `run()` class function, and accept `**kwargs` in their signature.

## AliasIngredient

::: blendsql.ingredients.ingredient.AliasIngredient
    handler: python
    show_source: true

## QAIngredient

::: blendsql.ingredients.ingredient.QAIngredient
    handler: python
    show_source: true

## MapIngredient

::: blendsql.ingredients.ingredient.MapIngredient
    handler: python
    show_source: true

## JoinIngredient

::: blendsql.ingredients.ingredient.JoinIngredient
    handler: python
    show_source: true
