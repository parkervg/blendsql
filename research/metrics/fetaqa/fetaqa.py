"""Spider concept metrics."""
from typing import Optional, Union
import datasets
from datasets import load_metric
from .evaluator import postprocess_text

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@article{Nan2021FeTaQAFT,
  title={FeTaQA: Free-form Table Question Answering},
  author={Nan, Linyong and Hsieh, Chiachun and Mao, Ziming and Lin, Xi Victoria and Verma, Neha and Zhang, Rui and Kryściński, Wojciech and Schoelkopf, Hailey and Kong, Riley and Tang, Xiangru and Mutuma, Mutethia and Rosand, Ben and Trindade, Isabel and Bandaru, Renusree and Cunningham, Jacob and Xiong, Caiming and Radev, Dragomir},
  journal={Transactions of the Association for Computational Linguistics},
  year={2022},
  volume={10},
  pages={35-49}
}
"""

_DESCRIPTION = """\
FeTaQA is a Free-form Table Question Answering dataset with 10K Wikipedia-based {table, 
question, free-form answer, supporting table cells} pairs. It yields a more challenging table 
QA setting because it requires generating free-form text answers after retrieval, inference,
and integration of multiple discontinuous facts from a structured knowledge source. 
Unlike datasets of generative QA over text in which answers are prevalent with copies of 
short text spans from the source, answers in our dataset are human-generated explanations 
involving entities and their high-level relations.
"""

_HOMEPAGE = "https://github.com/Yale-LILY/FeTaQA"

_LICENSE = "CC-BY-SA-4.0 License"


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION)
class FetaQAQuestion(datasets.Metric):
    def __init__(
        self,
        config_name: Optional[str] = None,
        keep_in_memory: bool = False,
        cache_dir: Optional[str] = None,
        num_process: int = 1,
        process_id: int = 0,
        seed: Optional[int] = None,
        experiment_id: Optional[str] = None,
        max_concurrent_cache_files: int = 10000,
        timeout: Union[int, float] = 100,
        **kwargs,
    ):
        super().__init__(
            config_name=config_name,
            keep_in_memory=keep_in_memory,
            cache_dir=cache_dir,
            num_process=num_process,
            process_id=process_id,
            seed=seed,
            experiment_id=experiment_id,
            max_concurrent_cache_files=max_concurrent_cache_files,
            timeout=timeout,
            **kwargs,
        )

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            features=datasets.Features(
                {
                    "predictions": datasets.features.Sequence(datasets.Value("string")),
                    "references": datasets.features.Features(
                        {
                            "answer_text": datasets.features.Sequence(
                                datasets.Value("string")
                            ),
                            "question": datasets.Value("string"),
                        }
                    ),
                }
            ),
            reference_urls=[""],
        )

    @staticmethod
    def eval_metric(preds, labels, metric_name="bertscore"):
        summary = {}
        if metric_name == "all":
            metric_list = ["sacrebleu", "rouge", "meteor", "bertscore", "bleurt"]
        else:
            metric_list = [metric_name]

        for metric_name in metric_list:
            metric = load_metric(metric_name)
            pred, gold = postprocess_text(preds, labels, metric_name)

            if metric_name == "bertscore":
                res = metric.compute(predictions=pred, references=gold, lang="en")
                for k, v in res.items():
                    if k == "hashcode":
                        continue
                    summary[f"{metric_name}_{k}"] = round(1.0 * sum(v) / len(v), 2)
            else:
                res = metric.compute(predictions=pred, references=gold)
                if metric_name == "sacrebleu":
                    summary[metric_name] = res["score"] * 0.01
                    # return res["score"] * 0.01  # limit it to range of [0, 1] for unifying
                elif metric_name == "bleurt":
                    summary["bleurt"] = round(
                        1.0 * sum(res["scores"]) / len(res["scores"]), 2
                    )
                    # return round(1.0 * sum(res["scores"]) / len(res["scores"]), 2)
                elif metric_name == "rouge":
                    for sub_metric_name in res.keys():
                        for i, key in enumerate(["precision", "recall", "fmeasure"]):
                            summary["{}_{}".format(sub_metric_name, key)] = res[
                                sub_metric_name
                            ][1][i]
                        # return res[sub_metric_name][1][-1]  #'fmeasure'
                    # this the the fmeasure('f-score') from the mid('mean aggregation')
                else:
                    summary[metric_name] = res[metric_name]
                    # return res[metric_name]
        return summary

    def _compute(self, predictions, references, allow_semantic: bool = True, **kwargs):
        assert len(predictions) == len(references)
        n_total_samples = len(predictions)
        n_correct_dict = None
        for pred, ref in zip(predictions, references):
            score = self.eval_metric(
                preds=pred, labels=ref["answer_text"], metric_name=self.config_name
            )
            if n_correct_dict:
                for key, value in score.items():
                    n_correct_dict[key] += value / n_total_samples
            else:
                n_correct_dict = {k: v / n_total_samples for k, v in score.items()}
        return n_correct_dict
