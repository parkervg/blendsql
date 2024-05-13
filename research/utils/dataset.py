import os
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
from pathlib import Path

from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from transformers.training_args import TrainingArguments

from .bridge_content_encoder import get_database_matches
from .args import ModelArguments

BASE_PATH = Path(os.path.dirname(os.path.realpath(__file__))) / ".."


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    clear_guidance_cache: bool = field(
        default=False,
        metadata={"help": "Clear internal guidance gptcache"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    val_max_time: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum allowed time in seconds for generation of one example. This setting can be used to stop "
            "generation whenever the full generation exceeds the specified amount of time."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )

    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )

    bypass_models: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Don't call models. For debugging, to get datapoints in predictions.json as fast as possible."
        },
    )
    fallback_to_prompt_and_pray: Optional[bool] = field(
        default=False,
        metadata={
            "help": "In the case of a bad BlendSQL output, fallback to prompt and pray answering."
        },
    )
    prompt_and_pray_only: Optional[bool] = field(
        default=False,
        metadata={"help": "Only do end-to-end table answering, no BlendSQL"},
    )
    include_doc_bridge_hints: Optional[bool] = field(
        default=False,
        metadata={"help": "Use vector cosine sim to include top docs in bridge hints"},
    )
    num_beams: int = field(
        default=1,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    num_beam_groups: int = field(
        default=1,
        metadata={
            "help": "Number of beam groups to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    diversity_penalty: Optional[float] = field(
        default=None,
        metadata={
            "help": "Diversity penalty to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    num_return_sequences: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of sequences to generate during evaluation. This argument will be passed to "
            "``model.generate``, which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None,
        metadata={
            "help": "A prefix to add before every source text (useful for T5 models)."
        },
    )
    schema_serialization_type: str = field(
        default="code",
        metadata={
            "help": "Choose between ``code`` and ``peteshaw`` schema serialization."
        },
    )
    schema_serialization_randomized: bool = field(
        default=False,
        metadata={"help": "Whether or not to randomize the order of tables."},
    )
    schema_serialization_with_db_id: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to add the database id to the context. Needed for Picard."
        },
    )
    schema_serialization_with_db_content: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to use the database content to resolve field matches."
        },
    )
    normalize_query: bool = field(
        default=True,
        metadata={
            "help": "Whether to normalize the SQL queries with the process in the 'Decoupling' paper"
        },
    )
    use_bridge_encoder: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to use Bridge Content Encoder during input serialization"
        },
    )
    db_path: Optional[List[str]] = field(
        default="research/db",
        metadata={"help": "Where to save temp SQLite databases"},
    )
    num_serialized_rows: Optional[int] = field(
        default=3,
        metadata={
            "help": "How many example rows to include in serialization of database"
        },
    )
    save_every: Optional[int] = field(
        default=50,
        metadata={"help": "Save results to predictions.json every n datapoints"},
    )
    truncate_content: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optionally limit serialized database value to this character length"
        },
    )

    schema_qualify: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to use sqlglot to qualify schema columns when calling `blend()`"
        },
    )

    def __post_init__(self):
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


@dataclass
class DataArguments:
    dataset: str = field(
        default="wikitq",
        metadata={"help": "The dataset to be used. Choose between `wikitq``."},
    )
    dataset_paths: Dict[str, str] = field(
        default_factory=lambda: {
            "wikitq": str(BASE_PATH / "./datasets/wikitq"),
            "hybridqa": str(BASE_PATH / "./datasets/hybridqa"),
            "feverous": str(BASE_PATH / "./datasets/feverous"),
            "ottqa": str(BASE_PATH / "./datasets/ottqa"),
            "fetaqa": str(BASE_PATH / "./datasets/fetaqa"),
        },
        metadata={"help": "Paths of the dataset modules."},
    )
    wikitq_dataset_url: str = field(
        default="",
        metadata={"help": "Path of wikitq.zip, relative to dataset path."},
    )
    squall_dataset_url: str = field(
        default="",
        metadata={"help": "Path of squall.zip, relative to dataset path."},
    )
    ottqa_dataset_url: str = field(
        default="",
        metadata={"help": "Path of ottqa.zip, relative to dataset path."},
    )
    fetaqa_dataset_url: str = field(
        default="",
        metadata={"help": "Path of squall.zip, relative to dataset path."},
    )

    metric_config: str = field(
        default="both",
        metadata={
            "help": "Choose between ``exact_match``, ``sacrebleu``,  '', or ``both``."
        },
    )
    long_answer: bool = field(
        default=False,
        metadata={"help": "whether or not should the model return long answer"},
    )
    metric_paths: Dict[str, str] = field(
        default_factory=lambda: {
            "wikitq": str(BASE_PATH / "./metrics/wikitq"),
            "hybridqa": str(BASE_PATH / "./metrics/hybridqa"),
            "feverous": str(BASE_PATH / "./metrics/feverous"),
            "ottqa": str(BASE_PATH / "./metrics/ottqa"),
            "fetaqa": str(BASE_PATH / "./metrics/fetaqa"),
        },
        metadata={"help": "Paths of the metric modules."},
    )
    data_config_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to data configuration file (specifying the database splits)"
        },
    )
    test_sections: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Sections from the data config to use for testing"},
    )


@dataclass
class TrainSplit(object):
    dataset: Dataset
    schemas: Dict[str, dict]


@dataclass
class EvalSplit(object):
    dataset: Dataset
    examples: Dataset


@dataclass
class DatasetSplits(object):
    train_split: Optional[TrainSplit]
    eval_split: Optional[EvalSplit]
    test_split: Optional[Dict[str, EvalSplit]]


def _prepare_eval_split(
    dataset: Dataset,
    data_training_args: DataTrainingArguments,
    model_args: ModelArguments,
    pre_process_function: Callable[[dict, Optional[int], Optional[int]], dict],
    max_example,
) -> EvalSplit:
    if max_example is not None and max_example < len(dataset):
        eval_examples = dataset.select(range(max_example))
    else:
        eval_examples = dataset
    eval_dataset = eval_examples.map(
        lambda batch: pre_process_function(
            batch=batch,
            data_training_args=data_training_args,
            model_args=model_args,
        ),
        batched=True,
        num_proc=data_training_args.preprocessing_num_workers,
        load_from_cache_file=not data_training_args.overwrite_cache,
    )
    return eval_dataset


def prepare_splits(
    dataset_dict: DatasetDict,
    data_args: DataArguments,
    training_args: TrainingArguments,
    data_training_args: DataTrainingArguments,
    model_args: ModelArguments,
    pre_process_function: Callable[[dict, Optional[int], Optional[int]], dict],
) -> DatasetSplits:
    train_split, eval_split, test_split = None, None, None

    if training_args.do_eval:
        eval_split = _prepare_eval_split(
            dataset_dict["validation"],
            data_training_args=data_training_args,
            model_args=model_args,
            pre_process_function=pre_process_function,
            max_example=data_training_args.max_val_samples,
        )

    if training_args.do_predict:
        test_split = _prepare_eval_split(
            dataset_dict["test"],
            data_training_args=data_training_args,
            model_args=model_args,
            pre_process_function=pre_process_function,
            max_example=data_training_args.max_test_samples,
        )

    if training_args.do_train:
        # For now, treat `train` like `validation`
        train_split = _prepare_eval_split(
            dataset_dict["train"],
            data_training_args=data_training_args,
            model_args=model_args,
            pre_process_function=pre_process_function,
            max_example=data_training_args.max_train_samples,
        )

    return DatasetSplits(
        train_split=train_split,
        eval_split=eval_split,
        test_split=test_split,
    )


def serialize_schema(
    question: str,
    db_path: str,
    db_id: str,
    db_column_names: Dict[str, str],
    db_table_names: List[str],
    schema_serialization_type: str = "peteshaw",
    schema_serialization_randomized: bool = False,
    schema_serialization_with_db_id: bool = True,
    schema_serialization_with_db_content: bool = False,
    normalize_query: bool = True,
    use_gold_concepts: bool = False,
    query: str = None,
) -> str:
    if use_gold_concepts and not query:
        raise ValueError(
            "If use_gold_concepts is True, need to pass gold SQL query as well"
        )
    if schema_serialization_type == "verbose":
        db_id_str = "Database: {db_id}. "
        table_sep = ". "
        table_str = "Table: {table}. Columns: {columns}"
        column_sep = ", "
        column_str_with_values = "{column} ({values})"
        column_str_without_values = "{column}"
        value_sep = ", "
    elif schema_serialization_type == "peteshaw":
        # see https://github.com/google-research/language/blob/master/language/nqg/tasks/spider/append_schema.py#L42
        db_id_str = " | {db_id}"
        table_sep = ""
        table_str = " | {table} : {columns}"
        column_sep = " , "
        column_str_with_values = "{column} ( {values} )"
        column_str_without_values = "{column}"
        value_sep = " , "
    else:
        raise NotImplementedError

    def get_column_str(
        table_name: str, column_name: str, gold_values: List[str] = None
    ) -> str:
        column_name_str = column_name.lower() if normalize_query else column_name
        if schema_serialization_with_db_content:
            if use_gold_concepts:
                # Encode the gold values from query
                if gold_values:
                    return column_str_with_values.format(
                        column=column_name_str, values=value_sep.join(gold_values)
                    )
                else:
                    return column_str_without_values.format(column=column_name_str)
            else:
                matches = get_database_matches(
                    question=question,
                    table_name=table_name,
                    column_name=column_name,
                    db_path=(db_path + "/" + db_id + "/" + db_id + ".sqlite"),
                )
                if matches:
                    return column_str_with_values.format(
                        column=column_name_str, values=value_sep.join(matches)
                    )
                else:
                    return column_str_without_values.format(column=column_name_str)
        else:
            return column_str_without_values.format(column=column_name_str)

    if use_gold_concepts:
        # Run SpiderSQL.to_gold_concepts to filter down schema
        # only to those concepts included in gold SQL
        ssql = SpiderSQL(
            data_dir="../data/spider/",
            db_path_fmt="database/{db_id}/{db_id}.sqlite",
        )
        try:
            items = ssql.to_gold_concepts(query, db_id=db_id)
            db_column_names = items.get("db_column_names")
            db_table_names = items.get("db_table_names")
        except:
            print(f"ERROR: {question}")
    else:
        # Just use the full 'db_column_names', 'db_table_names' we passed into this function
        pass

    tables = [
        table_str.format(
            table=table_name.lower() if normalize_query else table_name,
            columns=column_sep.join(
                map(
                    lambda y: get_column_str(
                        table_name=table_name, column_name=y[1], gold_values=y[2]
                    ),
                    filter(
                        lambda y: y[0] == table_id,
                        zip(
                            db_column_names["table_id"],
                            db_column_names["column_name"],
                            db_column_names.get(
                                "values", [None] * len(db_column_names["column_name"])
                            ),
                        ),
                    ),
                )
            ),
        )
        for table_id, table_name in enumerate(db_table_names)
    ]
    if schema_serialization_randomized:
        random.shuffle(tables)
    if schema_serialization_with_db_id:
        serialized_schema = db_id_str.format(db_id=db_id) + table_sep.join(tables)
    else:
        serialized_schema = table_sep.join(tables)
    return serialized_schema
