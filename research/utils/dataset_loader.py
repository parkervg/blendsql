import logging
from pathlib import Path

from datasets.arrow_dataset import Dataset
from datasets.metric import Metric
from typing import Tuple, Callable
from datasets.dataset_dict import DatasetDict
import datasets.load
from transformers.training_args import TrainingArguments

from .dataset import DataArguments, DataTrainingArguments, DatasetSplits, prepare_splits
from ..utils.args import ModelArguments
from .wikitq.wikitq import wikitq_pre_process_function, wikitq_metric_format_func
from .hybridqa.hybridqa import (
    hybridqa_pre_process_function,
    hybridqa_metric_format_func,
)
from .feverous.feverous import (
    feverous_pre_process_function,
    feverous_metric_format_func,
)
from .ottqa.ottqa import ottqa_pre_process_function, ottqa_metric_format_func

from .fetaqa.fetaqa import (
    fetaqa_pre_process_function,
    fetaqa_metric_format_func,
)

logger = logging.getLogger(__name__)


def _log_duplicate_count(dataset: Dataset, dataset_name: str, split: str) -> None:
    d = dataset.to_dict()
    d_t = [
        tuple((k, tuple(str(v))) for k, v in zip(d.keys(), vs))
        for vs in zip(*d.values())
    ]
    d_t_ = set(d_t)
    num_examples = len(d_t)
    duplicate_count = num_examples - len(d_t_)
    if duplicate_count > 0:
        logger.warning(
            f"The split ``{split}`` of the dataset ``{dataset_name}`` contains {duplicate_count} duplicates out of {num_examples} examples"
        )


def load_dataset(
    data_args: DataArguments,
    data_training_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
) -> Tuple[Metric, DatasetSplits]:
    # [dataset loader]
    _wikitq_dataset_dict: Callable[
        [], DatasetDict
    ] = lambda: datasets.load.load_dataset(
        path=data_args.dataset_paths["wikitq"],
        wikitq_dataset_url=data_args.wikitq_dataset_url,
        squall_dataset_url=data_args.squall_dataset_url,
    )
    _hybridqa_dataset_dict: Callable[
        [], DatasetDict
    ] = lambda: datasets.load.load_dataset(
        path=data_args.dataset_paths["hybridqa"],
    )
    _feverous_dataset_dict: Callable[
        [], DatasetDict
    ] = lambda: datasets.load.load_dataset(
        path=data_args.dataset_paths["feverous"],
    )
    _ottqa_dataset_dict: Callable[[], DatasetDict] = lambda: datasets.load.load_dataset(
        path=data_args.dataset_paths["ottqa"],
        db_output_dir=Path("research/db/ottqa").resolve(),
        ottqa_dataset_url=data_args.ottqa_dataset_url,
    )
    _fetaqa_dataset_dict: Callable[
        [], DatasetDict
    ] = lambda: datasets.load.load_dataset(
        path=data_args.dataset_paths["fetaqa"],
        fetaqa_dataset_url=data_args.fetaqa_dataset_url,
    )

    # [preprocessing func]
    _wikitq_pre_process_function = (
        lambda batch, data_training_args, model_args: wikitq_pre_process_function(
            batch=batch,
            data_training_args=data_training_args,
            model_args=model_args,
        )
    )
    _hybridqa_pre_process_function = (
        lambda batch, data_training_args, model_args: hybridqa_pre_process_function(
            batch=batch,
            data_training_args=data_training_args,
            model_args=model_args,
        )
    )
    _feverous_pre_process_function = (
        lambda batch, data_training_args, model_args: feverous_pre_process_function(
            batch=batch,
            data_training_args=data_training_args,
            model_args=model_args,
        )
    )
    _ottqa_pre_process_function = (
        lambda batch, data_training_args, model_args: ottqa_pre_process_function(
            batch=batch,
            data_training_args=data_training_args,
            model_args=model_args,
        )
    )
    _fetaqa_pre_process_function = (
        lambda batch, data_training_args, model_args: fetaqa_pre_process_function(
            batch=batch,
            data_training_args=data_training_args,
            model_args=model_args,
        )
    )

    # [dataset metric]
    _wikitq_metric: Callable[[], Metric] = lambda: datasets.load.load_metric(
        path=data_args.metric_paths["wikitq"],
        config_name=data_args.metric_config,
    )
    _hybridqa_metric: Callable[[], Metric] = lambda: datasets.load.load_metric(
        path=data_args.metric_paths["hybridqa"],
        config_name=data_args.metric_config,
    )
    _feverous_metric: Callable[[], Metric] = lambda: datasets.load.load_metric(
        path=data_args.metric_paths["feverous"],
        config_name=data_args.metric_config,
    )
    _ottqa_metric: Callable[[], Metric] = lambda: datasets.load.load_metric(
        path=data_args.metric_paths["ottqa"],
        config_name=data_args.metric_config,
    )
    _fetaqa_metric: Callable[[], Metric] = lambda: datasets.load.load_metric(
        path=data_args.metric_paths["fetaqa"],
        config_name=data_args.metric_config,
    )

    # [dataset metric format]
    _wikitq_metric_format_func: Callable = lambda item: wikitq_metric_format_func(
        item=item
    )
    _hybridqa_metric_format_func: Callable = lambda item: hybridqa_metric_format_func(
        item=item
    )
    _feverous_metric_format_func: Callable = lambda item: feverous_metric_format_func(
        item=item
    )
    _ottqa_metric_format_func: Callable = lambda item: ottqa_metric_format_func(
        item=item
    )
    _fetaqa_metric_format_func: Callable = lambda item: fetaqa_metric_format_func(
        item=item
    )

    _prepare_splits_kwargs = {
        "data_args": data_args,
        "training_args": training_args,
        "data_training_args": data_training_args,
        "model_args": model_args,
    }
    if data_args.dataset == "wikitq":
        metric = _wikitq_metric()
        dataset_splits = prepare_splits(
            dataset_dict=_wikitq_dataset_dict(),
            pre_process_function=_wikitq_pre_process_function,
            **_prepare_splits_kwargs,
        )
        metric_format_func = _wikitq_metric_format_func
    elif data_args.dataset == "hybridqa":
        metric = _hybridqa_metric()
        dataset_splits = prepare_splits(
            dataset_dict=_hybridqa_dataset_dict(),
            pre_process_function=_hybridqa_pre_process_function,
            **_prepare_splits_kwargs,
        )
        metric_format_func = _hybridqa_metric_format_func
    elif data_args.dataset == "feverous":
        metric = _feverous_metric()
        dataset_splits = prepare_splits(
            dataset_dict=_feverous_dataset_dict(),
            pre_process_function=_feverous_pre_process_function,
            **_prepare_splits_kwargs,
        )
        metric_format_func = _feverous_metric_format_func
    elif data_args.dataset == "ottqa":
        metric = _ottqa_metric()
        dataset_splits = prepare_splits(
            dataset_dict=_ottqa_dataset_dict(),
            pre_process_function=_ottqa_pre_process_function,
            **_prepare_splits_kwargs,
        )
        metric_format_func = _ottqa_metric_format_func
    elif data_args.dataset == "fetaqa":
        metric = _fetaqa_metric()
        dataset_splits = prepare_splits(
            dataset_dict=_fetaqa_dataset_dict(),
            pre_process_function=_fetaqa_pre_process_function,
            **_prepare_splits_kwargs,
        )
        metric_format_func = _fetaqa_metric_format_func
        import nltk

        nltk.download("punkt")
    else:
        raise ValueError(
            f"data_args.dataset {data_args.dataset} not currently supported!"
        )
    if dataset_splits.train_split is not None:
        _log_duplicate_count(
            dataset=dataset_splits.train_split,
            dataset_name=data_args.dataset,
            split="train",
        )
    if dataset_splits.eval_split is not None:
        _log_duplicate_count(
            dataset=dataset_splits.eval_split,
            dataset_name=data_args.dataset,
            split="eval",
        )
    if dataset_splits.test_split is not None:
        _log_duplicate_count(
            dataset=dataset_splits.test_split,
            dataset_name=data_args.dataset,
            split="test",
        )
    return metric, dataset_splits, metric_format_func
