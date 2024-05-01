# Set up logging
import logging
import sys
from typing import Tuple
from pathlib import Path
import sqlite3
import re

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

from ..database import to_serialized
from ..dataset import DataTrainingArguments
from ...utils.args import ModelArguments
from ...utils.bridge_content_encoder import (
    get_database_matches,
)
from ...constants import (
    SINGLE_TABLE_NAME,
    DOCS_TABLE_NAME,
    CREATE_VIRTUAL_TABLE_CMD,
    EvalField,
)
from ..normalizer import prepare_df_for_neuraldb_from_table
from blendsql.db import SQLite


def hybridqa_metric_format_func(item: dict) -> dict:
    prediction = item[EvalField.PREDICTION]
    if isinstance(prediction, str):
        prediction = [prediction]
    if prediction is not None:
        if len(prediction) < 1:
            pred = ""
        else:
            pred = prediction[0]
    else:
        pred = ""
    return {
        "prediction": str(pred),
        "reference": {
            "answer_text": item[EvalField.GOLD_ANSWER],
            "id": item[EvalField.UID],
            "question": item[EvalField.QUESTION],
        },
    }


def preprocess_hybridqa_table(table: dict) -> dict:
    """Preprocesses wikitq headers to make them easier to parse in text-to-SQL task.
    TODO: This is causing some encoding issues
    """
    preprocessed_table = {"header": [], "rows": []}
    for v in table["header"]:
        preprocessed_table["header"].append(re.sub(r"(\'|\")", "", v))
    for v in table["rows"]:
        preprocessed_table["rows"].append([re.sub(r"(\'|\")", "", item) for item in v])
    return preprocessed_table


def hybridqa_get_input(
    question: str,
    table: dict,
    passages: dict,
    table_id: str,
    data_training_args: DataTrainingArguments,
    model_args: ModelArguments,
) -> Tuple[str, dict]:
    """Prepares input for HybridQA dataset.

    Returns:
        Tuple containing:
            - str path to sqlite database
            - dict containing arguments to be passed to guidance program
    """
    db_path = Path(data_training_args.db_path) / "hybridqa" / f"{table_id}.db"
    if not db_path.is_file():
        # Create db
        if not db_path.parent.is_dir():
            db_path.parent.mkdir(parents=True)
        sqlite_conn = sqlite3.connect(db_path)
        prepare_df_for_neuraldb_from_table(
            preprocess_hybridqa_table(table), add_row_id=False
        ).to_sql(SINGLE_TABLE_NAME, sqlite_conn)
        # Create virtual table to search over
        c = sqlite_conn.cursor()
        c.execute(CREATE_VIRTUAL_TABLE_CMD)
        c.close()
        # Add content
        prepare_df_for_neuraldb_from_table(
            preprocess_hybridqa_table(passages), add_row_id=False
        ).to_sql(DOCS_TABLE_NAME, sqlite_conn, if_exists="append", index=False)
        sqlite_conn.close()
    db_path = str(db_path)
    db = SQLite(db_path)
    serialized_db = to_serialized(
        db=db,
        num_rows=data_training_args.num_serialized_rows,
    )
    entire_serialized_db = to_serialized(
        db=db,
        num_rows=data_training_args.num_serialized_rows,
        whole_table=True,
        truncate_content=data_training_args.truncate_content,
    )
    bridge_hints = None
    if data_training_args.use_bridge_encoder:
        bridge_hints = []
        column_str_with_values = "{table}.{column} ( {values} )"
        value_sep = " , "
        for table_name in db.iter_tables():
            if re.search(r"^{}_".format(DOCS_TABLE_NAME), table_name):
                continue
            for column_name in db.iter_columns(table_name):
                matches = get_database_matches(
                    question=question,
                    table_name=table_name,
                    column_name=column_name,
                    db_path=db_path,
                )
                if matches:
                    bridge_hints.append(
                        column_str_with_values.format(
                            table=table_name,
                            column=column_name,
                            values=value_sep.join(matches),
                        )
                    )
        bridge_hints = " , ".join(bridge_hints)
    db.con.close()
    return (
        db_path,
        {
            "few_shot_prompt": open("./research/prompts/hybridqa/few_shot.txt").read(),
            "ingredients_prompt": open(
                "./research/prompts/hybridqa/ingredients.txt"
            ).read(),
            "question": question,
            "serialized_db": serialized_db,
            "entire_serialized_db": entire_serialized_db,
            "bridge_hints": bridge_hints,
        },
    )


def hybridqa_pre_process_function(
    batch: dict, data_training_args: DataTrainingArguments, model_args: ModelArguments
) -> dict:
    db_path, input_program_args = zip(
        *[
            hybridqa_get_input(
                question=question,
                table=table,
                passages=passages,
                table_id=table_id,
                data_training_args=data_training_args,
                model_args=model_args,
            )
            for question, table, passages, table_id in zip(
                batch[EvalField.QUESTION],
                batch["table"],
                batch["passages"],
                batch["table_id"],
            )
        ]
    )
    return {"input_program_args": list(input_program_args), "db_path": list(db_path)}
