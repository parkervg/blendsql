"""Spider concept metrics."""

from typing import Optional, Union
import re
import datasets

try:
    from .evaluator import to_value_list, check_denotation
    from ...utils.normalizer import str_normalize
except:
    from research.metrics.wikitq.evaluator import to_value_list, check_denotation
    from research.utils.normalizer import str_normalize

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@inproceedings{pasupat-liang-2015-compositional,
    title = "Compositional Semantic Parsing on Semi-Structured Tables",
    author = "Pasupat, Panupong  and
      Liang, Percy",
    booktitle = "Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = jul,
    year = "2015",
    address = "Beijing, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P15-1142",
    doi = "10.3115/v1/P15-1142",
    pages = "1470--1480",
}
"""

_DESCRIPTION = """\
Two important aspects of semantic parsing for question answering are the breadth of the knowledge source and the depth of
logical compositionality. While existing work trades off one aspect for another, this paper simultaneously makes progress 
on both fronts through a new task: answering complex questions on semi-structured tables using question-answer pairs as 
supervision. The central challenge arises from two compounding factors: the broader domain results in an open-ended set 
of relations, and the deeper compositionality results in a combinatorial explosion in the space of logical forms. We 
propose a logical-form driven parsing algorithm guided by strong typing constraints and show that it obtains significant
 improvements over natural baselines. For evaluation, we created a new dataset of 22,033 complex questions on Wikipedia
  tables, which is made publicly available.
"""

_HOMEPAGE = "https://ppasupat.github.io/WikiTableQuestions/"

_LICENSE = "CC-BY-SA-4.0 License"


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION)
class WikiTableQuestion(datasets.Metric):
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
    def eval_ex_match(pred, gold, allow_semantic=True, question=None):
        if not isinstance(pred, list):
            pred = [pred]
            gold = [gold]

        pred = [str(p).lower().strip() for p in pred]
        gold = [str(g).lower().strip() for g in gold]
        if not allow_semantic:
            # WikiTQ eval w. string normalization using recognizer
            pred = [str_normalize(span) for span in pred]
            gold = [str_normalize(span) for span in gold]
            pred = to_value_list(pred)
            gold = to_value_list(gold)
            return check_denotation(pred, gold)
        else:
            assert isinstance(question, str)
            question = re.sub("\s+", " ", question).strip().lower()
            pred = [str_normalize(span) for span in pred]
            gold = [str_normalize(span) for span in gold]
            pred = sorted(list(set(pred)))
            gold = sorted(list(set(gold)))
            # (1) 0 matches 'no', 1 matches 'yes'; 0 matches 'more', 1 matches 'less', etc.
            if len(pred) == 1 and len(gold) == 1:
                if (pred[0] == "0" and gold[0] == "no") or (
                    pred[0] == "1" and gold[0] == "yes"
                ):
                    return True
                question_tokens = question.split()
                try:
                    pos_or = question_tokens.index("or")
                    token_before_or, token_after_or = (
                        question_tokens[pos_or - 1],
                        question_tokens[pos_or + 1],
                    )
                    if (pred[0] == "0" and gold[0] == token_after_or) or (
                        pred[0] == "1" and gold[0] == token_before_or
                    ):
                        return True
                except Exception:
                    pass
            # (2) Number value (allow units) and Date substring match
            if len(pred) == 1 and len(gold) == 1:
                NUMBER_UNITS_PATTERN = re.compile(
                    "^\$*[+-]?([0-9]*[.])?[0-9]+(\s*%*|\s+\w+)$"
                )
                DATE_PATTERN = re.compile(
                    "[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}\s*([0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2})?"
                )
                DURATION_PATTERN = re.compile("(P|PT)(\d+)(Y|M|D|H|S)")
                p, g = pred[0], gold[0]
                # Restore `duration` type, e.g., from 'P3Y' -> '3'
                if re.match(DURATION_PATTERN, p):
                    p = re.match(DURATION_PATTERN, p).group(2)
                if re.match(DURATION_PATTERN, g):
                    g = re.match(DURATION_PATTERN, g).group(2)
                match = False
                num_flag, date_flag = False, False
                # Number w. unit match after string normalization.
                # Either pred or gold being number w. units suffices it.
                if re.match(NUMBER_UNITS_PATTERN, p) or re.match(
                    NUMBER_UNITS_PATTERN, g
                ):
                    num_flag = True
                # Date match after string normalization.
                # Either pred or gold being date suffices it.
                if re.match(DATE_PATTERN, p) or re.match(DATE_PATTERN, g):
                    date_flag = True
                if num_flag:
                    p_set, g_set = set(p.split()), set(g.split())
                    if p_set.issubset(g_set) or g_set.issubset(p_set):
                        match = True
                if date_flag:
                    p_set, g_set = set(p.replace("-", " ").split()), set(
                        g.replace("-", " ").split()
                    )
                    if p_set.issubset(g_set) or g_set.issubset(p_set):
                        match = True
                if match:
                    return True
            pred = to_value_list(pred)
            gold = to_value_list(gold)
            return check_denotation(pred, gold)

    def _compute(self, predictions, references, allow_semantic: bool = True, **kwargs):
        assert len(predictions) == len(references)
        n_total_samples = len(predictions)
        n_correct_samples = 0
        for pred, ref in zip(predictions, references):
            score = self.eval_ex_match(
                pred=pred,
                gold=ref["answer_text"],
                allow_semantic=allow_semantic,
                question=ref["question"],
            )
            n_correct_samples += score
        return n_correct_samples / n_total_samples
