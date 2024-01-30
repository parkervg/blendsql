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
from ... import constants as CONST
from ...prompts.few_shot.hybridqa import blendsql_examples, sql_examples
from ..normalizer import prepare_df_for_neuraldb_from_table
from blendsql.db import SQLiteDBConnector


def hybridqa_metric_format_func(item: dict, flat_preds: list) -> dict:
    _pred = [i for i in flat_preds if i is not None]
    if len(_pred) < 1:
        pred = ""
    else:
        pred = _pred[0]
    return {
        "prediction": str(pred),
        "reference": {
            "answer_text": item["answer_text"],
            "id": item["id"],
            "question": item["question"],
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
        ).to_sql(CONST.SINGLE_TABLE_NAME, sqlite_conn)
        # Create virtual table to search over
        c = sqlite_conn.cursor()
        c.execute(CONST.CREATE_VIRTUAL_TABLE_CMD)
        c.close()
        # Add content
        prepare_df_for_neuraldb_from_table(
            preprocess_hybridqa_table(passages), add_row_id=False
        ).to_sql(CONST.DOCS_TABLE_NAME, sqlite_conn, if_exists="append", index=False)
        sqlite_conn.close()
    db_path = str(db_path)
    db = SQLiteDBConnector(db_path)
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
        doc_with_values = (
            "Sentence from `SELECT content FROM docs WHERE title = '{title}'`: '{sent}'"
        )
        value_sep = " , "
        for table_name in db.iter_tables():
            if re.search(r"^{}_".format(CONST.DOCS_TABLE_NAME), table_name):
                continue
            for column_name in db.iter_columns(table_name):
                # if (
                #     data_training_args.include_doc_bridge_hints
                #     and table_name == "docs"
                #     and column_name == "content"
                # ):
                #     matches = get_database_matches_docs(
                #         question=question,
                #         table_name=table_name,
                #         column_name=column_name,
                #         db_path=db_path,
                #     )
                #     if matches:
                #         for title, sent in matches:
                #             bridge_hints.append(
                #                 doc_with_values.format(title=title, sent=sent)
                #             )
                # else:
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
        bridge_hints = "\n".join(bridge_hints)
    db.con.close()
    return (
        db_path,
        {
            "examples": blendsql_examples
            if model_args.blender_model_name_or_path is not None
            else sql_examples,
            "question": question,
            "serialized_db": serialized_db,
            "entire_serialized_db": entire_serialized_db,
            "bridge_hints": bridge_hints,
            "extra_task_description": f"Additionally, we have the table `{CONST.DOCS_TABLE_NAME}` at our disposal, which contains Wikipedia articles providing more details about the values in our table.",
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
                batch["question"], batch["table"], batch["passages"], batch["table_id"]
            )
        ]
    )
    return {"input_program_args": list(input_program_args), "db_path": list(db_path)}
