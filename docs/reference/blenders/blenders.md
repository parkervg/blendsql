# Blenders

We use the term "blender" to describe the LLM which receives the prompts used to perform each ingredient function within a BlendSQL script.

We enable integration with many existing LLMs by building on top of `guidance` models: https://github.com/guidance-ai/guidance?tab=readme-ov-file#loading-models

Certain models may be better geared towards some BlendSQL tasks than others, so choose carefully!

::: blendsql.llms._llm.LLM
    handler: python
    show_source: true