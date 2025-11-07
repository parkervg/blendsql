---
hide:
  - toc
---
# Transformers

!!! Installation

    You need to install `llama-cpp-python` to use this in blendsql. 
    I used this command to install it on Ubuntu 25.10:
    ```
    CMAKE_ARGS="-DGGML_CUDA=ON -DLLAMA_LLAVA=OFF -DLLAVA_BUILD=OFF" uv pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
    ```


## TransformersLLM

::: blendsql.models.constrained.guidance.LlamaCpp
    handler: python
    show_source: false
