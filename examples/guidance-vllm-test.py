"""
vllm serve Qwen/Qwen3-4B-Instruct-2507 --host 0.0.0.0 \
--port 8000 \
--enable-prefix-caching \
--max-model-len 1028 \
--structured-outputs-config.backend guidance \
--attention-backend FLASHINFER \
--dtype bfloat16 \
--gpu_memory_utilization 0.8 \
--temperature 0.0
"""

import guidance

lm = guidance.models.experimental.VLLMModel(
    model="google/gemma-3-4b-it", base_url="http://localhost:8000/v1", api_key="N/A"
)

with guidance.user():
    lm += "Give me a Python list of 3 strings."
with guidance.assistant():
    # lm += "```python\nl = " + guidance.capture(gen_list(force_quotes=True, quantifier="{3}"), "response")
    lm += guidance.capture(guidance.select(["a", "b", "c"]), "response")
    # lm+= guidance.capture(guidance.regex("\d+"), "response")
print(str(lm))
print(lm["response"])
