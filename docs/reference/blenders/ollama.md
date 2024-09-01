---
hide:
  - toc
---
# Ollama

!!! Installation

    You need to install the `ollama` library to use this in blendsql.

!!! Note

    We consider Ollama models 'remote', since we're unable to access the underlying logits. As a result, we can only use Ollama for traditional generation, and not constrained generation (such as via the `options` arg in [LLMQA](../ingredients/LLMQA.md)) 

## OllamaLLM

::: blendsql.models.remote._ollama.OllamaLLM
    handler: python
    show_source: false