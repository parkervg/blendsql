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
from ... import constants as CONST
from ...prompts.few_shot.hybridqa import blendsql_examples, sql_examples
from ..normalizer import prepare_df_for_neuraldb_from_table
from blendsql.db import SQLiteDBConnector


def feverous_metric_format_func(item: dict, flat_preds: list) -> dict:
    _pred = [i for i in flat_preds if i is not None]
    if len(_pred) < 1:
        pred = ""
    else:
        pred = _pred[0]
    # Map `True` to 'SUPPORTS', `False` to 'REFUTES'
    pred = "SUPPORTS" if pred else "REFUTES"
    return {
        "prediction": str(pred),
        "reference": {"seq_out": item["label"]},
    }


def feverous_get_input(
    statement: str,
    table: dict,
    context: List[str],
    id: str,
    data_training_args: DataTrainingArguments,
    model_args: ModelArguments,
) -> Tuple[str, dict]:
    # Below id is unique for each datapoint
    # But, might be better to consider table_id instead
    db_path = Path(data_training_args.db_path) / "feverous" / f"{id}.db"
    tablename_to_description = {}
    if not db_path.is_file():
        # Create db
        if not db_path.parent.is_dir():
            db_path.parent.mkdir(parents=True)
        sqlite_conn = sqlite3.connect(db_path)
        for idx, (table_description, header, rows) in enumerate(
            zip(table["table_description"], table["header"], table["rows"])
        ):
            tablename = f"{CONST.SINGLE_TABLE_NAME}{idx}"
            prepare_df_for_neuraldb_from_table(
                {"header": header, "rows": rows}, add_row_id=False
            ).to_sql(tablename, sqlite_conn)
            tablename_to_description[tablename] = table_description
        if not all(len(x) == 0 for x in context.values()):
            # Create virtual table to search over
            c = sqlite_conn.cursor()
            c.execute(CONST.CREATE_VIRTUAL_TABLE_CMD)
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
            ).to_sql(
                CONST.DOCS_TABLE_NAME, sqlite_conn, if_exists="append", index=False
            )
            sqlite_conn.close()
    db_path = str(db_path)
    db = SQLiteDBConnector(db_path)
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
        doc_with_values = (
            "Sentence from `SELECT content FROM docs WHERE title = '{title}'`: '{sent}'"
        )
        value_sep = " , "
        for table_name in db.iter_tables():
            if re.search(r"^{}_".format(CONST.DOCS_TABLE_NAME), tablename):
                continue
            for column_name in db.iter_columns(table_name):
                # if (
                #     data_training_args.include_doc_bridge_hints
                #     and table_name == "docs"
                #     and column_name == "content"
                # ):
                #     matches = get_database_matches_docs(
                #         question=statement,
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
            "examples": blendsql_examples
            if model_args.blender_model_name_or_path is not None
            else sql_examples,
            "question": statement,
            "serialized_db": serialized_db,
            "entire_serialized_db": entire_serialized_db,
            "bridge_hints": bridge_hints,
            "extra_task_description": f"Additionally, we have the table `{CONST.DOCS_TABLE_NAME}` at our disposal, which contains Wikipedia articles providing more details about the values in our table.",
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
                id=id,
                data_training_args=data_training_args,
                model_args=model_args,
            )
            for statement, table, context, id in zip(
                batch["statement"], batch["table"], batch["context"], batch["id"]
            )
        ]
    )
    return {"input_program_args": list(input_program_args), "db_path": list(db_path)}
