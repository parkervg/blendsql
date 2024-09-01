---
hide:
  - toc
---
# Blenders

We use the term "blender" to describe the model which receives the prompts used to perform each ingredient function within a BlendSQL script.

We enable integration with many existing LLMs by building on top of [`guidance` models](https://github.com/guidance-ai/guidance).

Certain models may be better geared towards some BlendSQL tasks than others, so choose carefully!


## `Model`
::: blendsql.models._model.Model
    handler: python
    show_source: true