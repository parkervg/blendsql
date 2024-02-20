# Ingredients 

![ingredients](./img/
/ingredients.jpg)

Ingredients are at the core of a BlendSQL script. 

They are callable functions that perform one the task paradigms defined in [ingredient.py](./blendsql/ingredients/ingredient.py).

At their core, these are not a new concept. [User-defined functions (UDFs)](https://docs.databricks.com/en/udf/index.html), or [Application-Defined Functions in SQLite](https://www.sqlite.org/appfunc.html) have existed for quite some time. 

However, ingredients in BlendSQL are intended to be optimized towards LLM-based functions, defining an order of operations for traversing the AST such that the minimal amount of data is passed into your expensive GPT-4/Llama 2/Mistral 7b/etc. prompt.

Ingredient calls are denoted by wrapping them in double curly brackets, `{{ingredient}}`.
