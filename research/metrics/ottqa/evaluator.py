import re
import collections
import string
from ...utils.normalizer import str_normalize
from ..wikitq.evaluator import to_value_list, check_denotation

# copy from https://github.com/wenhuchen/OTT-QA/blob/master/evaluate_script.py


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def eval_ex_match(pred, gold, allow_semantic=True, question=None):
    """Taken from the Binder codebase, used to evaluate WikiTQ dataset."""
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
            if re.match(NUMBER_UNITS_PATTERN, p) or re.match(NUMBER_UNITS_PATTERN, g):
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


class EvaluateTool(object):
    def __init__(self, args=None):
        self.args = args

    def evaluate(self, preds, golds, section=None):
        summary = {}
        exact_scores = {}
        f1_scores = {}
        denotation_scores = {}
        for pred, gold in zip(preds, golds):
            qas_id = gold["id"]
            gold_answers = [gold["answer_text"]]

            exact_scores[qas_id] = max(compute_exact(a, pred) for a in gold_answers)
            f1_scores[qas_id] = max(compute_f1(a, pred) for a in gold_answers)
            denotation_scores[qas_id] = max(
                eval_ex_match(a, pred, question=gold["question"]) for a in gold_answers
            )
        total = len(golds)
        qid_list = list(exact_scores.keys())

        summary["exact"] = sum(exact_scores[k] for k in qid_list) / total
        summary["f1"] = sum(f1_scores[k] for k in qid_list) / total
        summary["denotation_acc"] = sum(denotation_scores[k] for k in qid_list) / total
        return summary
