from blendsql import blend, LLMMap, VQA
from blendsql.db import SQLite
from blendsql.models import TransformersLLM, VQAModel
import pandas as pd
import sqlite3
import re
import pandas as pd
from PIL import Image
import io
import os
import ast

TEST_QUERIES = [
    """
       SELECT {{VQA('What is in this image?', 'parks::Image')}} FROM parks
       WHERE "Location" = "Alaska"
    """,
    """
    SELECT {{VQA('What is in this image?', 'parks::Image')}} FROM parks
    WHERE "Location" = 'Alaska'
    ORDER BY {{
        LLMMap(
            'Size in km2?',
            'parks::Area'
        )
    }} LIMIT 2
    """,
]


def extract_path(md_link):
    return re.findall(r"\((.*?)\)", md_link)[0]


def to_bytes(path):
    try:
        # Open image and convert to bytes
        with Image.open(path, "r") as img:
            byte_arr = io.BytesIO()
            img.save(
                byte_arr, format="jpeg"
            )  # adjust format if your images are not JPEGs
            return byte_arr.getvalue()
    except Exception as e:
        print(f"Error: {e}")
        return None


def str_to_bytes(byte_string):
    byte_list = ast.literal_eval(byte_string)
    return bytes(byte_list)


if __name__ == "__main__":
    if not os.path.exists("example.db"):
        df = pd.read_csv("docs/img/national_parks_example/parks.csv")
        df["Image"] = df["Image"].apply(extract_path).apply(to_bytes)
        df["Image"] = df["Image"].apply(str_to_bytes)
        df["Location"] = df["Location"].apply(lambda x: x.strip())

        conn = sqlite3.connect("example.db")
        df.to_sql("parks", conn, if_exists="replace")

    db = SQLite("example.db")
    q = TEST_QUERIES[1]
    model = VQAModel("bczhou/tiny-llava-v1-hf")
    # Make our smoothie - the executed BlendSQL script
    smoothie = blend(
        query=q,
        db=db,
        blender=TransformersLLM("microsoft/Phi-3-mini-4k-instruct", caching=True),
        verbose=False,
        ingredients={LLMMap, VQA.from_args(model=model)},
    )
    print(smoothie.df)
