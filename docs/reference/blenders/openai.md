# OpenAI

!!! note

    In order to use this LLM as a Blender, we expect that you have a .env file created with all auth variables. 

## OpenaiLLM

::: blendsql.models.remote._openai.OpenaiLLM
    handler: python
    show_source: false
 
### Example Usage 
Given the following `.env` file in the current directory:
```text 
OPENAI_API_KEY=my_api_key
```

```python
from blendsql.models import OpenaiLLM

blender = OpenaiLLM("text-davinci-003", env=".")
```
## AzureOpenaiLLM

::: blendsql.models.remote._openai.AzureOpenaiLLM
    handler: python
    show_source: false

### Example Usage 
Given the following `.env` file in the current directory:
```text 
TENANT_ID=my_tenant_id
CLIENT_ID=my_client_id
CLIENT_SECRET=my_client_secret
```

```python
from blendsql.models import AzureOpenaiLLM

blender = AzureOpenaiLLM("text-davinci-003", env=".")
```