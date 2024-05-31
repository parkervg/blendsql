# Set up logging
import logging
import sys
from typing import Tuple, List
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
    CREATE_VIRTUAL_TABLE_CMD,
    DOCS_TABLE_NAME,
    EvalField,
)
from ..normalizer import prepare_df_for_neuraldb_from_table
from blendsql.db import SQLite


def feverous_metric_format_func(item: dict) -> dict:
    prediction = item[EvalField.PREDICTION]
    if prediction is not None:
        if len(prediction) < 1:
            pred = ""
        else:
            pred = prediction[0]
    else:
        pred = ""
    # Map `True` to 'SUPPORTS', `False` to 'REFUTES'
    pred = "SUPPORTS" if pred else "REFUTES"
    return {
        "prediction": str(pred),
        "reference": {"seq_out": item[EvalField.GOLD_ANSWER]},
    }


def feverous_get_input(
    statement: str,
    table: dict,
    context: List[str],
    uid: str,
    data_training_args: DataTrainingArguments,
    model_args: ModelArguments,
) -> Tuple[str, dict]:
    # Below uid is unique for each datapoint
    # But, might be better to consider table_id instead
    db_path = Path(data_training_args.db_path) / "feverous" / f"{uid}.db"
    tablename_to_description = {}
    contains_documents = not all(len(x) == 0 for x in context.values())
    if not db_path.is_file():
        # Create db
        if not db_path.parent.is_dir():
            db_path.parent.mkdir(parents=True)
        sqlite_conn = sqlite3.connect(db_path)
        for idx, (table_description, header, rows) in enumerate(
            zip(table["table_description"], table["header"], table["rows"])
        ):
            tablename = f"{SINGLE_TABLE_NAME}{idx}"
            prepare_df_for_neuraldb_from_table(
                {"header": header, "rows": rows}, add_row_id=False
            ).to_sql(tablename, sqlite_conn)
            tablename_to_description[tablename] = table_description
        if contains_documents:
            # Create virtual table to search over
            c = sqlite_conn.cursor()
            c.execute(CREATE_VIRTUAL_TABLE_CMD)
            c.close()
            # Add content
            prepare_df_for_neuraldb_from_table(
                {
                    "header": ["title", "content"],
                    "rows": [
                        [title, content]
                        for title, content in set(
                            tuple(zip(context["title"], context["content"]))
                        )
                    ],
                },
                add_row_id=False,
            ).to_sql(DOCS_TABLE_NAME, sqlite_conn, if_exists="append", index=False)
            sqlite_conn.close()
    db_path = str(db_path)
    db = SQLite(db_path)
    serialized_db = to_serialized(
        db=db,
        num_rows=data_training_args.num_serialized_rows,
        tablename_to_description=tablename_to_description,
    )
    entire_serialized_db = to_serialized(
        db=db,
        num_rows=data_training_args.num_serialized_rows,
        tablename_to_description=tablename_to_description,
        whole_table=True,
        truncate_content=300,
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
                    question=statement,
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
            "few_shot_prompt": open("./research/prompts/feverous/few_shot.txt").read(),
            "ingredients_prompt": open(
                "./research/prompts/feverous/ingredients.txt"
            ).read(),
            "question": statement,
            "serialized_db": serialized_db,
            "entire_serialized_db": entire_serialized_db,
            "bridge_hints": bridge_hints,
            "extra_task_description": (
                f"Additionally, we have the table `{DOCS_TABLE_NAME}` at our disposal, which contains Wikipedia articles providing more details about the values in our table."
                if contains_documents
                else ""
            ),
        },
    )


def feverous_pre_process_function(
    batch: dict, data_training_args: DataTrainingArguments, model_args: ModelArguments
) -> dict:
    db_path, input_program_args = zip(
        *[
            feverous_get_input(
                statement=statement,
                table=table,
                context=context,
                uid=uid,
                data_training_args=data_training_args,
                model_args=model_args,
            )
            for statement, table, context, uid in zip(
                batch[EvalField.QUESTION],
                batch["table"],
                batch["context"],
                batch[EvalField.UID],
            )
        ]
    )
    return {"input_program_args": list(input_program_args), "db_path": list(db_path)}
