# Creating Custom BlendSQL Ingredients

All the built-in LLM ingredients inherit from the base classes `QAIngredient`, `MapIngredient`, and `JoinIngredient`.

These are intended to be helpful abstractions, so that the user can easily implement their own functions to run within a BlendSQL script.

The processing logic for a custom ingredient should go in a `run()` class function, and accept `**kwargs` in their signature.

## QAIngredient

::: blendsql.ingredients.ingredient.QAIngredient.run
    handler: python
    show_source: true

### Example:

```python
import pandas as pd
from blendsql.ingredients import QAIngredient
from blendsql._program import Program, gen


class SummaryProgram(Program):
    """Program to call LLM and return summary of the passed table.
    """

    def __call__(self, serialized_db: str):
        with self.usercontext:
            self.model += f"Summarize the table below.\n\n{serialized_db}\n"
        with self.assistantcontext:
            self.model += gen(name="result", **self.gen_kwargs)
        return self.model


class TableSummary(QAIngredient):
    def run(self, context: pd.DataFrame, llm: 'LLM', **kwargs) -> str:
        result = llm.predict(program=SummaryProgram, serialized_db=context.to_string())["result"]
        return f"'{result}'"


if __name__ == "__main__":
    from blendsql import blend
    from blendsql.db import SQLiteDBConnector
    from blendsql.utils import fetch_from_hub
    from blendsql.llms import OpenaiLLM

    blendsql = """
    {{
        TableSummary(
            context=(SELECT * FROM transactions LIMIT 10)
        )
    }}
    """

    smoothie = blend(
        query=blendsql,
        blender=OpenaiLLM("gpt-4"),
        db=SQLiteDBConnector(fetch_from_hub("single_table.db")),
        ingredients={TableSummary}
    )
```

Returns:

```
'The table shows a list of expenses made through Zelle, an online payment service. The expenses range from $175 to $2000. All transactions are categorized under "Cash/ATM" as the parent category. The dates of these transactions span throughout the year 2022.'
```

## MapIngredient

::: blendsql.ingredients.ingredient.MapIngredient.run
    handler: python
    show_source: true

### Example:

```python
from typing import List
from blendsql.ingredients import MapIngredient
import requests

class GetQRCode(MapIngredient):
    """Calls API to generate QR code for a given URL.
    Saves bytes to file in qr_codes/ and returns list of paths.
    https://goqr.me/api/doc/create-qr-code/
    """
    def run(self, values: List[str], **kwargs) -> List[str]:
        imgs_as_bytes = []
        for value in values:
            qr_code_bytes = requests.get(
                "https://api.qrserver.com/v1/create-qr-code/?data=https://{}/&size=100x100".format(value)
            ).content
            imgs_as_bytes.append(qr_code_bytes)
        return imgs_as_bytes

if __name__ == "__main__":
    from blendsql import blend
    from blendsql.db import SQLiteDBConnector
    from blendsql.utils import fetch_from_hub
    
    blendsql = "SELECT genre, url, {{GetQRCode('QR Code as Bytes:', 'w::url')}} FROM w WHERE genre = 'social'"
    
    smoothie = blend(
        query=blendsql,
        blender=None,
        db=SQLiteDBConnector(fetch_from_hub("urls.db")),
        ingredients={GetQRCode}
    )
```

Returns:

| genre  | url           | QR Code as Bytes:      |
|--------|---------------|-----------------------|
| social | facebook.com  | b'\x89PNG\r\n\x1a\n\x00\x00...  |
| social | instagram.com | b'\x89PNG\r\n\x1a\n\x00\x00... |
| social | tiktok.com    | b'\x89PNG\r\n\x1a\n\x00\x00...    |

## JoinIngredient

::: blendsql.ingredients.ingredient.JoinIngredient.run
    handler: python
    show_source: true