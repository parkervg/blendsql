from typing import Optional, Union
import datasets

try:
    from .evaluator import EvaluateTool
except:
    from research.metrics.ottqa.evaluator import EvaluateTool

_CITATION = """\
@article{chen2020open,
  title={Open question answering over tables and text},
  author={Chen, Wenhu and Chang, Ming-Wei and Schlinger, Eva and Wang, William and Cohen, William W},
  journal={arXiv preprint arXiv:2010.10439},
  year={2020}
}
"""

_DESCRIPTION = """\
This dataset is obtained from the official release of the OTT-QA.
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION)
class OTTQA(datasets.Metric):
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
        **kwargs
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
            **kwargs
        )
        self.evaluator = EvaluateTool()

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": datasets.features.Features(
                        {
                            "answer_text": datasets.Value("string"),
                            "id": datasets.Value("string"),
                            "question": datasets.Value("string"),
                        }
                    ),
                }
            ),
            reference_urls=[""],
        )

    def _compute(self, predictions, references, **kwargs):
        assert len(predictions) == len(references)
        return self.evaluator.evaluate(preds=predictions, golds=references)
