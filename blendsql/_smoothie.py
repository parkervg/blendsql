import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Any
import inspect
import subprocess

import pandas as pd
from nbformat.v4.nbbase import (
    new_code_cell,
    new_markdown_cell,
    new_notebook,
    new_output,
)
from .ingredients import Ingredient

"""
Defines output of an executed BlendSQL script
"""


@dataclass
class SmoothieMeta:
    process_time_seconds: float
    num_values_passed: int  # Number of values passed to a Map ingredient
    example_map_outputs: List[
        Any
    ]  # 10 example outputs from a Map ingredient, for debugging
    ingredients: List[Ingredient]
    query: str
    db_path: str
    contains_ingredient: bool = True


@dataclass
class Smoothie:
    df: pd.DataFrame
    meta: SmoothieMeta

    def save_recipe(
        self, output_file: str, recipe_name: str = None, as_html: bool = False
    ) -> None:
        """Use nbformat to write recipe context to a Jupyter notebook.

        Args:
            output_file: Filepath to save the ipynb/html output to
            recipe_name: Heading for the Jupyter notebook. Defaults to output_file.stem
            as_html: Boolean to optionally save notebook as html as well.

        """
        output_file = Path(output_file)
        recipe_name = recipe_name if recipe_name is not None else output_file.stem
        cells = [
            new_markdown_cell(source=f"# {recipe_name}"),
            new_code_cell(
                source="from blendsql import blend, SQLiteDBConnector\nfrom blendsql.ingredients import MapIngredient, LLM, DT",
                execution_count=1,
            ),
            new_code_cell(
                source=f'db = SQLiteDBConnector("{self.meta.db_path}")',
                execution_count=2,
            ),
            new_markdown_cell(source="## Ingredients"),
        ]
        for idx, ingredient in enumerate(self.meta.ingredients):
            cells.append(
                new_code_cell(
                    source=inspect.getsource(ingredient), execution_count=idx + 3
                )
            )
        cells.append(new_markdown_cell("## Query"))
        cells.append(
            new_code_cell(
                f'query = """{self.meta.query}"""',
                execution_count=idx + 4,
                metadata={"collapsed": False},
            )
        )
        cells.append(
            new_code_cell(
                source="""smoothie = blend(\n   query=query,\n   db=db,\n   ingredients=ingredients\n)\nsmoothie.df""",
                execution_count=idx + 5,
                outputs=[
                    new_output(
                        output_type="display_data",
                        data={
                            "text/plain": self.df.to_string(),
                            "text/html": self.df.to_html(),
                        },
                    )
                ],
            )
        )
        nb = new_notebook(cells=cells, metadata={"language": "python"})
        with open(output_file, "w") as f:
            json.dump(nb.dict(), f)
        if as_html:
            subprocess.run(
                f"jupyter nbconvert --to html {output_file}",
                shell=True,
                env=dict(os.environ),
            )
