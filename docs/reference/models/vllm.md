---
hide:
  - toc
---

## `VLLM`

To begin, start a vLLM server. Be sure to specify `--structured-outputs-config.backend guidance` if your vLLM version is `>0.12.0`. 

```
vllm serve RedHatAI/gemma-3-12b-it-quantized.w4a16 --host 0.0.0.0 \
--port 8000 \
--enable-prefix-caching \
--max-model-len 8000 \
--structured-outputs-config.backend guidance \
--gpu_memory_utilization 0.8 \
--enable-prompt-tokens-details
```

::: blendsql.models.vllm.VLLM
    handler: python
    show_source: true