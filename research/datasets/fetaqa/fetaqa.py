# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors, The Google AI Language Team Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO： This code can be push to HuggingFace as a new contribution.
"""FeTaQA, a Free-form Table Question Answering dataset"""
import os
import json
import datasets

from research.constants import EvalField

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@article{nan2021fetaqa,
  title={FeTaQA: Free-form Table Question Answering},
  author={Nan, Linyong and Hsieh, Chiachun and Mao, Ziming and Lin, Xi Victoria and Verma, Neha and Zhang, Rui and Kry{\'s}ci{\'n}ski, Wojciech and Schoelkopf, Nick and Kong, Riley and Tang, Xiangru and others},
  journal={arXiv preprint arXiv:2104.00369},
  year={2021}
}
"""

_DESCRIPTION = """\
FeTaQA is a Free-form Table Question Answering dataset with 10K Wikipedia-based
 {table, question, free-form answer, supporting table cells} pairs. It yields a
more challenging table QA setting because it requires generating free-form text
 answers after retrieval, inference, and integration of multiple discontinuous 
facts from a structured knowledge source. Unlike datasets of generative QA over
 text in which answers are prevalent with copies of short text spans from the source,
  answers in our dataset are human-generated explanations involving entities and their
   high-level relations.

This dataset is obtained from the official release of the FETAQA.
"""

_HOMEPAGE = "https://github.com/Yale-LILY/FeTaQA/"

_LICENSE = "CC-BY-SA-4.0 License"

_URL = "https://github.com/Yale-LILY/FeTaQA/raw/main/data/"
_TRAINING_FILE = "fetaQA-v1_train.jsonl"
_DEV_FILE = "fetaQA-v1_dev.jsonl"
_TEST_FILE = "fetaQA-v1_test.jsonl"


class FETAQA(datasets.GeneratorBasedBuilder):
    """The FETAQA dataset"""

    def __init__(
        self, *args, writer_batch_size=None, fetaqa_dataset_url=_URL, **kwargs
    ) -> None:
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self._url = fetaqa_dataset_url

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    EvalField.UID: datasets.Value("int32"),
                    EvalField.QUESTION: datasets.Value("string"),
                    "table_id": datasets.Value("string"),
                    "table": {
                        "header": datasets.features.Sequence(datasets.Value("string")),
                        "rows": datasets.features.Sequence(
                            datasets.features.Sequence(datasets.Value("string"))
                        ),
                    },
                    "meta": datasets.Value("string"),
                    EvalField.GOLD_ANSWER: datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_files = dl_manager.download_and_extract(self._url)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(downloaded_files, _TRAINING_FILE)},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(downloaded_files, _DEV_FILE)},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(downloaded_files, _TEST_FILE)},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                example = json.loads(line)
                yield idx, {
                    EvalField.UID: example["feta_id"],
                    EvalField.QUESTION: example["question"],
                    "table_id": example["table_source_json"],
                    "table": {
                        "header": example["table_array"][0],
                        "rows": example["table_array"][1:],
                    },
                    "meta": example["table_page_title"]
                    + " | "
                    + example["table_section_title"],
                    EvalField.GOLD_ANSWER: example["answer"],
                }
