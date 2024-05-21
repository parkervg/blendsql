import logging
import sys
from typing import Tuple
from pathlib import Path
import sqlite3

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

from ..dataset import DataTrainingArguments
from ...utils.args import ModelArguments
from ...prompts.few_shot.fetaqa import blendsql_examples, sql_examples


from ...utils.bridge_content_encoder import get_database_matches
from ...constants import SINGLE_TABLE_NAME, EvalField
from ..normalizer import prepare_df_for_neuraldb_from_table
from blendsql.db import SQLite


def fetaqa_metric_format_func(item: dict) -> dict:
    prediction = item.get(EvalField.PREDICTION, None)
    if prediction is not None:
        if len(prediction) < 1:
            pred = ""
        else:
            pred = prediction[0]
    else:
        pred = ""
    return {
        "prediction": [str(pred)],
        "reference": {
            "answer_text": [item["answer_text"]],
            "question": item["question"],
        },
    }


def fetaqa_get_input(
    question: str,
    title: dict,
    table: str,
    table_id: str,
    data_training_args: DataTrainingArguments,
    model_args: ModelArguments,
) -> Tuple[str, dict]:
    """Prepares input for WikiTableQuestions dataset.

    Returns:
        Tuple containing:
            - str path to sqlite database
            - dict containing arguments to be passed to guidance program
    """
    # table_id in format csv/204-csv/772.csv
    table_id = Path(table_id)
    db_path = (
        Path(data_training_args.db_path)
        / "fetaqa"
        / table_id.parent
        / f"{table_id.stem}.db"
    )
    if not db_path.is_file():
        # Create db
        if not db_path.parent.is_dir():
            db_path.parent.mkdir(parents=True)
        sqlite_conn = sqlite3.connect(db_path)
        prepare_df_for_neuraldb_from_table(table, add_row_id=False).to_sql(
            SINGLE_TABLE_NAME, sqlite_conn
        )
    db_path = str(db_path)
    db = SQLite(db_path)
    serialized_db = db.to_serialized(
        num_rows=data_training_args.num_serialized_rows,
        table_description=title,
    )
    bridge_hints = None
    if data_training_args.use_bridge_encoder:
        bridge_hints = []
        column_str_with_values = "{column} ( {values} )"
        value_sep = " , "
        for table_name in db.iter_tables():
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
                            column=column_name, values=value_sep.join(matches)
                        )
                    )
        bridge_hints = "\n".join(bridge_hints)
    db.con.close()
    return (
        db_path,
        {
            "examples": (
                blendsql_examples
                if model_args.blender_model_name_or_path is not None
                else sql_examples
            ),
            "question": question,
            "serialized_db": serialized_db,
            "bridge_hints": bridge_hints,
            "extra_task_description": "Provide concrete reasoning to the answer",
        },
    )


def fetaqa_pre_process_function(
    batch: dict, data_training_args: DataTrainingArguments, model_args: ModelArguments
) -> dict:
    db_path, input_program_args = zip(
        *[
            fetaqa_get_input(
                question=question,
                title=title,
                table=table,
                table_id=table_id,
                data_training_args=data_training_args,
                model_args=model_args,
            )
            for question, table, title, table_id in zip(
                batch[EvalField.QUESTION],
                batch["table"],
                batch["meta"],
                batch["table_id"],
            )
        ]
    )
    return {"input_program_args": list(input_program_args), "db_path": list(db_path)}
