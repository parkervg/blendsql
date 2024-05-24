---
hide:
  - toc
---
### Smoothie 
The [smoothie.py](./blendsql/_smoothie.py) object defines the output of an executed BlendSQL script.

```python
@dataclass
class Smoothie:
    df: pd.DataFrame
    meta: SmoothieMeta
    
@dataclass
class SmoothieMeta:
    process_time_seconds: float
    num_values_passed: int  # Number of values passed to a Map/Join/QA ingredient
    num_prompt_tokens: int  # Number of prompt tokens (counting user and assistant, i.e. input/output)
    prompts: List[str] # Log of prompts submitted to model
    example_map_outputs: List[Any]  # outputs from a Map ingredient, for debugging
    ingredients: List[Ingredient]
    query: str
    db_path: str
    contains_ingredient: bool = True

def blend(*args, **kwargs) -> Smoothie:
  ... 
```