# Set up logging
import logging
import sys
from typing import Tuple
from functools import lru_cache

from typing import List
from rank_bm25 import BM25Okapi

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
from ... import constants as CONST
from ...prompts.few_shot.ottqa import blendsql_examples

from blendsql.db import SQLiteDBConnector

#db = SQLiteDBConnector("./research/db/ottqa/ottqa.db")


@lru_cache
def vectorize_texts(model, texts: Tuple[str]):
    return model.encode(texts, convert_to_tensor=True, show_progress_bar=True)


def ottqa_metric_format_func(item: dict, flat_preds: list) -> dict:
    _pred = [i for i in flat_preds if i is not None]
    if len(_pred) < 1:
        pred = ""
    else:
        pred = _pred[0]
    return {
        "prediction": str(pred),
        "reference": {"answer_text": item["answer_text"], "id": item["id"]},
    }


def tokenize(s: str) -> List[str]:
    return s.split(" ")
    # return re.findall("\w+|[^\w\s]+", s)


def ottqa_get_input(
    question: str,
    db_path: str,
    id: str,
    data_training_args: DataTrainingArguments,
    model_args: ModelArguments,
) -> Tuple[str, dict]:
    all_tables = tuple(
        [i for i in db.iter_tables() if not i.startswith(CONST.DOCS_TABLE_NAME)]
    )
    all_tokenized_tables = [tokenize(i) for i in all_tables]
    bm25 = BM25Okapi(all_tokenized_tables)
    chosen_tables = [
        " ".join(choice)
        for choice in bm25.get_top_n(tokenize(question), all_tokenized_tables, n=5)
    ]
    serialized_db = to_serialized(
        db=db,
        num_rows=data_training_args.num_serialized_rows,
        use_tables=chosen_tables,
        truncate_content=500,
    )

    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # question_vec = vectorize_texts(model, [question])
    # table_vec = vectorize_texts(model, all_tables)
    # sim = util.cos_sim(table_vec, question_vec)
    # chosen_tables = [
    #     all_tables[index] for index in sim.squeeze().argsort(descending=True)[:10]
    # ]
    # db.con.close()
    return (
        db_path,
        {
            "examples": blendsql_examples
            if model_args.blender_model_name_or_path is not None
            else None,
            "question": question,
            "serialized_db": serialized_db,
            "entire_serialized_db": None,
            "bridge_hints": None,
            "use_tables": chosen_tables,
            "extra_task_description": f"Additionally, we have the table `{CONST.DOCS_TABLE_NAME}` at our disposal, which contains Wikipedia articles providing more details about the values in our table.",
        },
    )


def ottqa_pre_process_function(
    batch: dict, data_training_args: DataTrainingArguments, model_args: ModelArguments
) -> dict:
    db_path, input_program_args = zip(
        *[
            ottqa_get_input(
                question=question,
                db_path=db_path,
                id=id,
                data_training_args=data_training_args,
                model_args=model_args,
            )
            for question, db_path, id in zip(
                batch["question"], batch["db_path"], batch["id"]
            )
        ]
    )
    return {"input_program_args": list(input_program_args), "db_path": list(db_path)}
