from typing import Optional, Union
import datasets

try:
    from .evaluator import EvaluateTool
except:
    from research.metrics.hybridqa.evaluator import EvaluateTool

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@article{chen2020hybridqa,
  title={HybridQA: A Dataset of Multi-Hop Question Answering over Tabular and Textual Data},
  author={Chen, Wenhu and Zha, Hanwen and Chen, Zhiyu and Xiong, Wenhan and Wang, Hong and Wang, William},
  journal={Findings of EMNLP 2020},
  year={2020}
}
"""

_DESCRIPTION = """\
Existing question answering datasets focus on dealing with homogeneous information, based either only on text or KB/Table information alone. However, as human knowledge is distributed over heterogeneous forms, using homogeneous information alone might lead to severe coverage problems. To fill in the gap, we present HybridQA, a new large-scale question-answering dataset that requires reasoning on heterogeneous information. Each question is aligned with a Wikipedia table and multiple free-form corpora linked with the entities in the table. The questions are designed to aggregate both tabular information and text information, i.e., lack of either form would render the question unanswerable. We test with three different models: 1) a table-only model. 2) text-only model. 3) a hybrid model that combines heterogeneous information to find the answer. The experimental results show that the EM scores obtained by two baselines are below 20%, while the hybrid model can achieve an EM over 40%. This gap suggests the necessity to aggregate heterogeneous information in HybridQA. However, the hybrid model’s score is still far behind human performance. Hence, HybridQA can serve as a challenging benchmark to study question answering with heterogeneous information.
"""

_HOMEPAGE = "https://hybridqa.github.io/"

_LICENSE = "CC-BY-SA-4.0 License"


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION)
class HybridQA(datasets.Metric):
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
        return self.evaluator.evaluate(predictions, references)
