import copy
import os
import logging
import sys
import shutil

import sqlglot
from colorama import Fore
import re
import textwrap

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

import sys
from pathlib import Path
import json
from tqdm import tqdm
from dataclasses import asdict
import guidance
import time
import numpy as np
from typing import List, Union, Callable
from attr import attrs, attrib
from sqlglot import parse_one, exp

import datasets
from datasets.metric import Metric
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.hf_argparser import HfArgumentParser

from blendsql.db import SQLiteDBConnector
from blendsql.db.utils import double_quote_escape
from blendsql import LLMMap, LLMQA, LLMJoin, LLMValidate, blend
from blendsql.utils import TokenTimer, sub_tablename
from blendsql.ingredients.builtin.llm.utils import initialize_endpoint
from blendsql.ingredients.builtin.llm.endpoint import Endpoint
from blendsql._constants import OPENAI_CHAT_LLM
from blendsql._dialect import FTS5SQLite
from blendsql._smoothie import Smoothie
from blendsql._grammar import grammar

from research.utils.dataset import DataArguments, DataTrainingArguments
from research.utils.dataset_loader import load_dataset
from research.utils.args import ModelArguments
from research.constants import SINGLE_TABLE_NAME, EvalField
from research.prompts import programs


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def choose_parser_program(model_args: ModelArguments):
    """Depending on ModelArguments, choose the appropriate parser program template."""
    using_blendsql = bool(model_args.blender_model_name_or_path is not None)
    if model_args.parser_model_name_or_path in OPENAI_CHAT_LLM:
        if using_blendsql:
            program_template = programs.fewshot_blendsql_program_chat
        else:
            program_template = programs.fewshot_sql_program_chat
    else:
        if using_blendsql:
            program_template = programs.fewshot_blendsql_program_completion
        else:
            program_template = programs.fewshot_sql_program_completion
    return textwrap.dedent(program_template)


def fewshot_parse_to_blendsql(
    endpoint: "Endpoint", program: str, **input_program_args
) -> str:
    """Calls an endpoint_name and generates a BlendSQL query."""
    # Dedent str args
    for k, v in input_program_args.items():
        if isinstance(v, str):
            input_program_args[k] = textwrap.dedent(v)
    res = endpoint.predict(program=program, **input_program_args)
    return textwrap.dedent(res["result"])


def post_process_blendsql(blendsql: str, db: SQLiteDBConnector, use_tables=None) -> str:
    """Clean up some common mistakes made by LLM parser.
    This includes:
        - Aligning hallucinated columns to their closest match in the database
        - Wrapping all column references in double quotes
            - ONLY if it's not already within quotes (', ")
    """

    def parse_str_and_add_columns(
        s: str, valid_columns: set, real_colname_to_hallucinated: dict
    ):
        try:
            node = parse_one(s, dialect=FTS5SQLite)
            for n in node.find_all(exp.Column):
                if n.name not in valid_columns:
                    split_on_underscores = " ".join(n.name.split("_"))
                    if split_on_underscores in valid_columns:
                        real_colname_to_hallucinated[split_on_underscores] = n.name
        except sqlglot.ParseError:
            pass
        return real_colname_to_hallucinated

    if use_tables is None:
        use_tables = set(
            filter(lambda x: not x.startswith("documents"), list(db.iter_tables()))
        )

    blendsql = blendsql.replace("`", "'")
    blendsql = blendsql.replace("{{LLM(", "{{LLMMap(")
    # Below fixes case where we miss a ')'
    # SELECT MAX({{LLMMap('total penalties?', 'w::penalties (p+p+s+s)'}}) FROM w
    blendsql = re.sub(r"((MAX|MIN)\(\{\{.*?)(\'}}\))", r"\1')}})", blendsql)
    blendsql = re.sub("'}}", "')}}", blendsql)
    # Handle mistakes like {{LLMMap('field goal percentage?'; 'w::field goal\xa0%')}}
    blendsql = re.sub(r"(\'|\"); ", r"\1,", blendsql)
    quotes_start_end = [i.start() for i in re.finditer(r"(\'|\")", blendsql)]
    quotes_start_end_spans = list(zip(*(iter(quotes_start_end),) * 2))

    # Find some hallucinated column names
    flatten = lambda xss: set([x for xs in xss for x in xs])
    valid_columns = flatten(
        [list(i) for i in list(db.iter_columns(table) for table in use_tables)]
    )
    real_colname_to_hallucinated = {}
    real_colname_to_hallucinated = parse_str_and_add_columns(
        blendsql, valid_columns, real_colname_to_hallucinated
    )
    for parse_results, _, _ in grammar.scanString(blendsql):
        parsed_results_dict = parse_results.as_dict()
        for arg_type in {"args", "kwargs"}:
            for idx in range(len(parsed_results_dict[arg_type])):
                curr_arg = parsed_results_dict[arg_type][idx]
                if not isinstance(curr_arg, str):
                    continue
                parsed_results_dict[arg_type][idx] = re.sub(
                    r"(^\()(.*)(\)$)", r"\2", curr_arg
                ).strip()
        potential_subquery = re.sub(
            r"JOIN(\s+){{.+}}", "", parsed_results_dict["args"][1], flags=re.DOTALL
        )
        real_colname_to_hallucinated = parse_str_and_add_columns(
            potential_subquery, valid_columns, real_colname_to_hallucinated
        )

    for k, v in real_colname_to_hallucinated.items():
        blendsql = sub_tablename(v, k, blendsql)

    for tablename in db.iter_tables(use_tables=use_tables):
        for columnname in sorted(
            list(db.iter_columns(tablename)), key=lambda x: len(x), reverse=True
        ):
            # Reverse finditer so we don't mess up indices when replacing
            # Only sub if surrounded by: whitespace, comma, or parentheses
            # Or, prefaced by period (e.g. 'p.Current_Value')
            # AND it's not already in quotes
            for m in list(
                re.finditer(
                    r"(?<=(\s|,|\(|\.)){}(?=(\s|,|\)|;|$))".format(
                        re.escape(columnname)
                    ),
                    blendsql,
                )
            )[::-1]:
                # Check if m.start already occurs within quotes (' or ")
                # If it does, don't add quotes
                if any(
                    start + 1 < m.start() < end
                    for (start, end) in quotes_start_end_spans
                ):
                    continue
                blendsql = (
                    blendsql[: m.start()]
                    + '"'
                    + double_quote_escape(
                        blendsql[m.start() : m.start() + (m.end() - m.start())]
                    )
                    + '"'
                    + blendsql[m.end() :]
                )
    return blendsql


@attrs
class BlendSQLEvaluation:
    output_dir: Union[str, Path] = attrib()
    split: datasets.Split = attrib()
    split_name: str = attrib()
    parser_endpoint: Union[Endpoint, None] = attrib()
    blender_endpoint: Union[Endpoint, None] = attrib()
    prompt_and_pray_endpoint: Union[Endpoint, None] = attrib()
    parser_program: guidance.Program = attrib()
    model_args: ModelArguments = attrib()
    data_args: DataArguments = attrib()
    data_training_args: DataTrainingArguments = attrib()
    db: SQLiteDBConnector = attrib(default=None)

    results: List[dict] = attrib(init=False)
    results_dict: dict = attrib(init=False)
    num_with_ingredients: int = attrib(init=False)
    num_errors: int = attrib(init=False)

    def __attrs_post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.results = []
        self.num_with_ingredients = 0
        self.num_errors = 0

    def _init_results_dict(self):
        return {
            EvalField.UID: None,
            "dataset_vars": None,
            "idx": None,
            "input_program_args": None,
            EvalField.DB_PATH: None,
            EvalField.PRED_BLENDSQL: None,
            "num_few_shot_examples": None,
            EvalField.PREDICTION: [""],
            EvalField.GOLD_ANSWER: None,
            "solver": None,
            "error": None,
        }

    def iter_eval(self):
        logger.info("*** Evaluate ***")
        logger.info(f"--- {len(self.split)} examples ---")
        for _idx, item in tqdm(
            enumerate(self.split),
            desc=f"Running over {self.split_name}_split...",
            total=len(self.split),
        ):
            if (_idx % self.data_training_args.save_every) == 0:
                print(
                    Fore.WHITE + "Saving predictions.json as checkpoint..." + Fore.RESET
                )
                with open(self.output_dir / "predictions.json", "w") as f:
                    json.dump(self.results, f, indent=4, cls=NpEncoder)
            self.results_dict = self._init_results_dict()
            _item = copy.deepcopy(item)
            for v in [
                value
                for name, value in vars(EvalField).items()
                if not name.startswith("_")
            ]:
                if v in _item:
                    self.results_dict[v] = _item.pop(v)
            self.results_dict["dataset_vars"] = {
                k: v
                for k, v in _item.items()
                if k not in {"passages", "table", "input_program_args"}
            }
            self.results_dict["idx"] = _idx
            entire_serialized_db = None
            if "entire_serialized_db" in item["input_program_args"]:
                entire_serialized_db = item["input_program_args"].pop(
                    "entire_serialized_db"
                )
            self.results_dict["input_program_args"] = {
                k: v
                for k, v in item["input_program_args"].items()
                if k not in {"examples", "program", "endpoint_name"}
            }
            self.results_dict["num_few_shot_examples"] = len(
                item["input_program_args"]["examples"]
            )
            self.results_dict[EvalField.DB_PATH] = item[EvalField.DB_PATH]
            if self.db is None:
                db = SQLiteDBConnector(item[EvalField.DB_PATH])
            else:
                db = self.db
            if not self.data_training_args.bypass_models:
                if not self.data_training_args.prompt_and_pray_only:
                    pred_text = self._get_blendsql_prediction(item, db)
                    if self.data_training_args.fallback_to_prompt_and_pray:
                        # Fallback to end-to-end QA prompt
                        if (
                            any(x in pred_text for x in ["table", "passage", "text"])
                            and any(x in pred_text for x in ["not", "empty"])
                        ) or pred_text.strip() == "":
                            if entire_serialized_db is None:
                                raise ValueError(
                                    "Trying to fallback to end-to-end, but no `entire_serialized_db` variable found!"
                                )
                            _ = self._get_prompt_and_pray_prediction(
                                item, entire_serialized_db
                            )
                elif self.data_training_args.prompt_and_pray_only:
                    _ = self._get_prompt_and_pray_prediction(item, entire_serialized_db)
            self.results.append(self.results_dict)
            # Log predictions to console
            print()
            print(Fore.MAGENTA + item[EvalField.QUESTION] + Fore.RESET)
            if self.results_dict[EvalField.PRED_BLENDSQL] is not None:
                print(
                    Fore.CYAN + self.results_dict[EvalField.PRED_BLENDSQL] + Fore.RESET
                )
            print(
                Fore.MAGENTA + f"ANSWER: '{self.results_dict[EvalField.GOLD_ANSWER]}'"
            )
            if self.results_dict[EvalField.PREDICTION] is not None:
                print(
                    Fore.CYAN
                    + str(self.results_dict[EvalField.PREDICTION])
                    + Fore.RESET
                )
            print()
        with open(self.output_dir / "predictions.json", "w") as f:
            json.dump(self.results, f, indent=4, cls=NpEncoder)

    def _get_prompt_and_pray_prediction(self, item: dict, entire_serialized_db: str):
        try:
            to_add = {"solver": "prompt-and-pray"}
            res = self.prompt_and_pray_endpoint.predict(
                program=programs.zero_shot_qa_program_chat,
                question=item["input_program_args"]["question"],
                serialized_db=entire_serialized_db,
            )
            final_str_pred: str = [res.get("result", "")]
            to_add[EvalField.PREDICTION] = final_str_pred
            self.results_dict = self.results_dict | to_add
        except Exception as error:
            print(Fore.RED + "Error in get_prompt_and_pray prediction" + Fore.RESET)
            print(Fore.RED + str(error) + Fore.RESET)
            self.results_dict = self.results_dict | to_add
            return [""]

    def _get_blendsql_prediction(self, item: dict, db: SQLiteDBConnector) -> List[str]:
        to_add = {"solver": "blendsql"}
        try:
            blendsql = fewshot_parse_to_blendsql(
                endpoint=self.parser_endpoint,
                program=self.parser_program,
                **item["input_program_args"],
            )
            blendsql = post_process_blendsql(
                blendsql=blendsql,
                db=db,
                use_tables=item["input_program_args"].get("use_tables", None),
            )
            to_add[EvalField.PRED_BLENDSQL] = blendsql
            res: Smoothie = blend(
                query=blendsql,
                db=db,
                ingredients={LLMMap, LLMQA, LLMJoin, LLMValidate}
                if self.model_args.blender_model_name_or_path is not None
                else set(),
                # Force usage of the endpoint_name we specify
                overwrite_args={
                    "endpoint": self.blender_endpoint,
                    "long_answer": self.data_args.long_answer,
                },
                table_to_title={
                    SINGLE_TABLE_NAME: item["table"].get("page_title", None)
                },
                infer_map_constraints=True,
                silence_db_exec_errors=False,
                verbose=True,
            )
            pred_has_ingredient = res.meta.contains_ingredient
            self.num_with_ingredients += pred_has_ingredient
            to_add["pred_has_ingredient"] = pred_has_ingredient
            to_add["example_map_outputs"] = res.meta.example_map_outputs
            prediction = [i for i in res.df.values.flat if i is not None]
            to_add[EvalField.PREDICTION] = prediction
            self.results_dict = self.results_dict | to_add
            return prediction
        except Exception as error:
            print(Fore.RED + "Error in get_blendsql prediction" + Fore.RESET)
            print(Fore.RED + str(error) + Fore.RESET)
            self.results_dict = self.results_dict | to_add
            self.results_dict["error"] = str(error)
            return [""]

    def save_metrics(self, metric: Metric, metric_format_func: Callable):
        # Finally, read from predictions.json and calculate metrics
        with open(self.output_dir / "predictions.json", "r") as f:
            predictions = json.load(f)
        for item in predictions:
            metric.add(**metric_format_func(item | item["dataset_vars"]))
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(
                {
                    "metric_score": metric.compute(),
                    "num_with_ingredients": self.num_with_ingredients,
                    "num_errors": self.num_errors,
                    "num_completed": len(self.results),
                    "split_size": len(self.split),
                },
                f,
                indent=4,
            )
        combined_args_dict = {
            **asdict(self.model_args),
            **asdict(self.data_args),
            **asdict(self.data_training_args)
            # **training_args.to_sanitized_dict(),
        }

        with open(self.output_dir / "combined_args.json", "w") as f:
            json.dump(combined_args_dict, f, indent=4)

        print(Fore.GREEN + f"Saved outputs to {self.output_dir}" + Fore.RESET)


def main() -> None:
    time.time()
    # See all possible arguments by passing the --help flag to this script.
    parser = HfArgumentParser(
        (DataArguments, DataTrainingArguments, Seq2SeqTrainingArguments, ModelArguments)
    )
    data_args: DataArguments
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        (
            data_args,
            data_training_args,
            training_args,
            model_args,
        ) = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]), allow_extra_keys=True
        )
    elif (
        len(sys.argv) == 3
        and sys.argv[1].startswith("--local_rank")
        and sys.argv[2].endswith(".json")
    ):
        data = json.loads(Path(os.path.abspath(sys.argv[2])).read_text())
        data.update({"local_rank": int(sys.argv[1].split("=")[1])})
        (data_args, data_training_args, training_args, model_args) = parser.parse_dict(
            args=data
        )
    else:
        (
            data_args,
            data_training_args,
            training_args,
            model_args,
        ) = parser.parse_args_into_dataclasses()
    if data_training_args.clear_guidance_cache:
        guidance.llms.OpenAI.cache.clear()
    if data_training_args.overwrite_cache:
        # Remove the appropriate directory containing our save db files
        if data_args.dataset == "wikitq":
            dataset_db_path = Path(data_training_args.db_path) / "wikitq"
            if dataset_db_path.is_dir():
                shutil.rmtree(str(dataset_db_path))

    # Load dataset
    metric, dataset_splits, metric_format_func = load_dataset(
        data_args=data_args,
        data_training_args=data_training_args,
        model_args=model_args,
        training_args=training_args,
    )
    if (
        not training_args.do_train
        and not training_args.do_eval
        and not training_args.do_predict
    ):
        logger.info(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`."
        )
        return
    output_dir = Path(training_args.output_dir)
    # cached_idx = 0
    # prev_predictions = None
    # predictions_json_filename = output_dir / "predictions.json"
    # if predictions_json_filename.is_file() and training_args.overwrite_output_dir:
    #     # Load from cache
    #     print(Fore.YELLOW + f"Attempting to load checkpoint from predictions.json cache at {str(predictions_json_filename)}..." + Fore.RESET)
    #     with open(predictions_json_filename, "r") as f:
    #         prev_predictions = json.load(f)
    #     cached_idx = max([item["idx"] for item in prev_predictions]) if prev_predictions != [] else 0
    #     print(Fore.GREEN + f"Starting inference with checkpoint idx {cached_idx}." + Fore.RESET)
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)
    elif not training_args.overwrite_output_dir:
        raise ValueError("output_dir is not empty, and overwrite_output_dir is False!")

    parser_endpoint = initialize_endpoint(model_args.parser_model_name_or_path)
    parser_program = choose_parser_program(model_args)
    parser_endpoint.gen_kwargs["temperature"] = model_args.parser_temperature
    if data_training_args.bypass_models:
        parser_endpoint.predict = lambda *args, **kwargs: {"result": "SELECT TRUE;"}
    blender_endpoint = None

    if model_args.blender_model_name_or_path is not None:
        blender_endpoint = initialize_endpoint(model_args.blender_model_name_or_path)
        blender_endpoint.gen_kwargs["temperature"] = model_args.blender_temperature
        if data_training_args.bypass_models:
            blender_endpoint.predict = lambda *args, **kwargs: {"result": ""}

    prompt_and_pray_endpoint = None
    if model_args.prompt_and_pray_model_name_or_path is not None:
        prompt_and_pray_endpoint = initialize_endpoint(
            model_args.prompt_and_pray_model_name_or_path
        )

    splits = {}
    if training_args.do_eval:
        splits["eval"] = dataset_splits.eval_split
    elif training_args.do_train:
        splits["train"] = dataset_splits.train_split
    elif training_args.do_predict:
        splits["test"] = dataset_splits.test_split

    if data_args.dataset == "ottqa":
        # Load the massive db only once
        db = SQLiteDBConnector("./research/db/ottqa/ottqa.db")
    else:
        db = None

    for curr_split_name, curr_split in splits.items():
        bse = BlendSQLEvaluation(
            split=curr_split,
            split_name=curr_split_name,
            output_dir=output_dir,
            parser_endpoint=parser_endpoint,
            blender_endpoint=blender_endpoint,
            prompt_and_pray_endpoint=prompt_and_pray_endpoint,
            parser_program=parser_program,
            model_args=model_args,
            data_args=data_args,
            data_training_args=data_training_args,
            db=db,
        )
        try:
            bse.iter_eval()
        except Exception as error:
            raise error
        finally:
            bse.save_metrics(metric=metric, metric_format_func=metric_format_func)


if __name__ == "__main__":
    # Start a process running in the background to refresh our OpenAI token
    timer = TokenTimer("./secrets.json", refresh_interval_min=30)
    timer.start()
    time.sleep(5)
    main()
