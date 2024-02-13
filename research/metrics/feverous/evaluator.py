import numpy as np


class EvaluateTool(object):
    def __init__(self, args=None):
        self.args = args

    def evaluate(self, preds, golds, section=None):
        summary = {}
        all_match = []

        for pred, gold_item in zip(preds, golds):
            # IMPORTANT!
            # Below we ignore "NOT ENOUGH INFO"
            # Consider this when comparing to other results
            if gold_item["seq_out"] == "NOT ENOUGH INFO":
                continue
            match_or_not = pred == gold_item["seq_out"]
            all_match.append(match_or_not)

        summary["all"] = float(np.mean(all_match))

        return summary
