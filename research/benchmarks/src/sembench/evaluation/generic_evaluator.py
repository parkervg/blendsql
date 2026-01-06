"""
Created on June 2, 2025

@author: Jiale Lao

Generic evaluator base class for all use cases, you should implement
"generate_ground_truth" and "evaluate" functions for each query.
The function signatures for each query qi should be:
- self._generate_qi_ground_truth() -> pd.DataFrame:
- self._evaluate_qi()
- self._evaluate_qi()
"""

from __future__ import annotations

from dataclasses import dataclass
from cdlib import NodeClustering, evaluation

import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import f1_score


@dataclass
class QueryMetricRetrieval:
    """Metrics for retrieval tasks (e.g., finding relevant items)."""

    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0


@dataclass
class QueryMetricAggregation:
    """Metrics for aggregation tasks (e.g., counting, summing)."""

    relative_error: float = 0.0
    absolute_error: float = 0.0
    mean_absolute_percentage_error: float = 0.0

    def calculate_errors(self, predicted: float, actual: float) -> None:
        """Populate the error fields based on *predicted* vs *actual*."""
        self.absolute_error = float(abs(predicted - actual))
        if actual != 0:
            self.relative_error = float(self.absolute_error / abs(actual))
            self.mean_absolute_percentage_error = float(self.relative_error * 100)
        else:
            self.relative_error = float("inf") if predicted != 0 else 0.0
            self.mean_absolute_percentage_error = float(self.relative_error * 100)


@dataclass
class QueryMetricRank:
    """Metrics for ranking tasks (e.g., scoring and ranking items)."""

    spearman_correlation: float = 0.0
    kendall_tau: float = 0.0


@dataclass
class SingleAccuracyScore:
    """
    Accuracy score class that can be used for any type of metric that produces a single float value.

    Makes post-processing easier, e.g., plotting, because it's transparent to the type of metric.
    """

    accuracy: float
    metric_type: str


@dataclass
class SingleAccuracyScoreWithRetrievalDetails(SingleAccuracyScore):
    """
    Special case for retrieval tasks that can contain more details on the accuracy metric.
    Not suitable for Adjusted-Rand-Index or Omega-Index
    """

    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0


class GenericEvaluator:
    """Abstract base class for benchmark evaluators."""

    def _discover_ground_truth_impl(self, query_id) -> callable:
        method_name = f"_generate_q{query_id}_ground_truth"
        try:
            query_fn = getattr(self, method_name)
            if not callable(query_fn):
                raise TypeError(f"{method_name} exists but is not callable")
            return query_fn
        except AttributeError:
            raise NotImplementedError(
                f"Query {query_id} not implemented for {self.system_name}."
            ) from None

    def _discover_evaluate_impl(self, query_id) -> callable:
        method_name = f"_evaluate_q{query_id}"
        try:
            query_fn = getattr(self, method_name)
            if not callable(query_fn):
                raise TypeError(f"{method_name} exists but is not callable")
            return query_fn
        except AttributeError:
            raise NotImplementedError(
                f"Query {query_id} not implemented for {self.system_name}."
            ) from None

    def _generic_retrieval_evaluation(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricRetrieval:
        """
        Generic evaluation for retrieval queries WITHOUT limit clauses.

        Compares system results with ground truth and calculates precision, recall, and F1.
        """

        if len(ground_truth) == 0:
            return QueryMetricRetrieval(
                precision=1.0 if len(system_results) == 0 else 0.0
            )
        if len(system_results) == 0:
            return QueryMetricRetrieval()

        matches = 0
        matched_gt = set()
        for _, srow in system_results.iterrows():
            for gt_idx, gt_row in ground_truth.iterrows():
                if gt_idx in matched_gt:
                    continue
                common = set(srow.index) & set(gt_row.index)
                if all(
                    srow[c] == gt_row[c]
                    for c in common
                    if pd.notna(srow[c]) and pd.notna(gt_row[c])
                ):
                    matches += 1
                    matched_gt.add(gt_idx)
                    break
        precision = matches / len(system_results)
        recall = matches / len(ground_truth)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        return QueryMetricRetrieval(precision, recall, f1)

    def _generic_aggregation_evaluation(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricAggregation:
        """
        Generic evaluation for aggregation queries.

        Assumes single value comparison. Handles string-to-numeric conversion.
        """

        m = QueryMetricAggregation()
        if len(system_results) != 1 or len(ground_truth) != 1:
            m.relative_error = 1.0
            m.absolute_error = float("inf")
            m.mean_absolute_percentage_error = 100.0
            return m

        def first_num(df):
            for c in df.columns:
                val = df[c].iloc[0]

                # Try to convert to numeric if it's a string
                if isinstance(val, str):
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        continue

                # Check if it's numeric after potential conversion
                if pd.api.types.is_numeric_dtype(type(val)) or isinstance(
                    val, (int, float)
                ):
                    return val
            return None

        sys_val, gt_val = first_num(system_results), first_num(ground_truth)
        if sys_val is None or gt_val is None:
            (
                m.relative_error,
                m.absolute_error,
                m.mean_absolute_percentage_error,
            ) = (1.0, float("inf"), 100.0)
            return m

        m.calculate_errors(predicted=sys_val, actual=gt_val)
        return m

    def _evaluate_tuple_matching(
        self,
        system_results: pd.DataFrame,
        ground_truth: pd.DataFrame,
        n_columns: int,
    ) -> QueryMetricRetrieval:
        """
        Generic function to evaluate queries that require matching tuples of n
        columns. Works for pairs (n=2), triples (n=3), quadruples (n=4), etc.
        """

        # Fast both-empty check
        if system_results.empty and ground_truth.empty:
            return QueryMetricRetrieval(1.0, 1.0, 1.0)

        # If one is empty, handle per convention
        if system_results.empty and not ground_truth.empty:
            return QueryMetricRetrieval()
        if ground_truth.empty and not system_results.empty:
            return QueryMetricRetrieval()

        # Check if we have enough columns
        if len(system_results.columns) < n_columns:
            return QueryMetricRetrieval()

        # Use first n columns regardless of their names
        sys_cols = list(system_results.columns[:n_columns])
        gt_cols = list(ground_truth.columns[:n_columns])

        def create_tuple(row, cols):
            values = [row[col] for col in cols]
            if all(pd.notna(v) for v in values):
                return tuple(sorted(values))
            return None

        # Create sets of tuples from both dataframes
        sys_tuples = {
            t
            for t in system_results.apply(lambda r: create_tuple(r, sys_cols), axis=1)
            if t is not None
        }
        gt_tuples = {
            t
            for t in ground_truth.apply(lambda r: create_tuple(r, gt_cols), axis=1)
            if t is not None
        }

        # Calculate metrics
        correct = sys_tuples & gt_tuples
        precision = len(correct) / len(sys_tuples) if sys_tuples else 0.0
        recall = len(correct) / len(gt_tuples) if gt_tuples else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        return QueryMetricRetrieval(precision, recall, f1)

    def _evaluate_unique_values(
        self,
        system_results: pd.DataFrame,
        ground_truth: pd.DataFrame,
        column_index: int = 0,
    ) -> QueryMetricRetrieval:
        """
        Generic function to evaluate queries that compare unique values from a
        specific column.

        Used for queries like Q5, Q7, Q8, Q9, Q10 that compare owner names or
        similar.
        """

        # Fast both-empty check
        if system_results.empty and ground_truth.empty:
            return QueryMetricRetrieval(1.0, 1.0, 1.0)

        # If one is empty, handle per convention
        if system_results.empty and not ground_truth.empty:
            return QueryMetricRetrieval()
        if ground_truth.empty and not system_results.empty:
            return QueryMetricRetrieval()

        # Check if we have enough columns
        if len(system_results.columns) < column_index:
            return QueryMetricRetrieval()

        # Use the column at the specified index (default: first column)
        sys_col = system_results.columns[column_index]
        gt_col = (
            ground_truth.columns[column_index]
            if len(ground_truth.columns) > column_index
            else ground_truth.columns[0]
        )

        # Get unique values from both sets
        system_values = set(system_results[sys_col].dropna())
        ground_truth_values = set(ground_truth[gt_col].dropna())

        # Calculate true positives
        tp = len(system_values & ground_truth_values)

        # Calculate metrics
        precision = (
            (tp / len(system_values))
            if system_values
            else (1.0 if not ground_truth_values else 0.0)
        )
        recall = (
            (tp / len(ground_truth_values))
            if ground_truth_values
            else (1.0 if not system_values else 0.0)
        )
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        return QueryMetricRetrieval(precision, recall, f1_score)

    def _generic_ranking_evaluation(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricRank:
        """
        Generic evaluation for ranking queries.

        Assumes first column is the id and second column is the score/rank.
        Calculates Spearman's rank correlation coefficient and Kendall's tau coefficient.
        """
        from scipy.stats import spearmanr, kendalltau

        if len(system_results) == 0 or len(ground_truth) == 0:
            return QueryMetricRank(spearman_correlation=0.0, kendall_tau=0.0)

        # Ensure we have at least 2 columns (id, score)
        if len(system_results.columns) < 2 or len(ground_truth.columns) < 2:
            return QueryMetricRank(spearman_correlation=0.0, kendall_tau=0.0)

        # Use first column as id, second as score
        sys_id_col = system_results.columns[0]
        sys_score_col = system_results.columns[1]
        gt_id_col = ground_truth.columns[0]
        gt_score_col = ground_truth.columns[1]

        # Create dictionaries for mapping id to score
        sys_scores = {}
        for _, row in system_results.iterrows():
            id_val = row[sys_id_col]
            score_val = row[sys_score_col]
            if pd.notna(id_val) and pd.notna(score_val):
                try:
                    sys_scores[id_val] = float(score_val)
                except (ValueError, TypeError):
                    continue

        gt_scores = {}
        for _, row in ground_truth.iterrows():
            id_val = row[gt_id_col]
            score_val = row[gt_score_col]
            if pd.notna(id_val) and pd.notna(score_val):
                try:
                    gt_scores[id_val] = float(score_val)
                except (ValueError, TypeError):
                    continue

        # Find common IDs
        common_ids = set(sys_scores.keys()) & set(gt_scores.keys())
        if len(common_ids) < 2:
            return QueryMetricRank(spearman_correlation=0.0, kendall_tau=0.0)

        # Create aligned arrays for correlation calculation
        sys_values = [sys_scores[id_val] for id_val in common_ids]
        gt_values = [gt_scores[id_val] for id_val in common_ids]

        # Calculate correlations
        try:
            spearman_result = spearmanr(sys_values, gt_values)
            spearman_corr = (
                spearman_result.correlation
                if not pd.isna(spearman_result.correlation)
                else 0.0
            )
        except Exception:
            spearman_corr = 0.0

        try:
            kendall_result = kendalltau(sys_values, gt_values)
            kendall_corr = (
                kendall_result.correlation
                if not pd.isna(kendall_result.correlation)
                else 0.0
            )
        except Exception:
            kendall_corr = 0.0

        return QueryMetricRank(
            spearman_correlation=spearman_corr.item(), kendall_tau=kendall_corr.item()
        )

    def compute_precision(
        ground_truth: pd.DataFrame,
        query_result: pd.DataFrame,
        id_column: str = "id",
    ):
        """
        Computes the precision of the query result towards the ground truth.
        Assumes that both dataframes have an "id" column the uniquely identifies
        a row. The precision is computed based on the IDs in the result and the
        ground truth.
        """
        ground_truth_ids = (
            set(ground_truth[id_column]) if not ground_truth.empty else set()
        )
        result_ids = (
            set(query_result[id_column])
            if query_result is not None and not query_result.empty
            else set()
        )
        if len(ground_truth_ids) == 0:
            # If ground truth is empty, precision is 1.0 if result is also
            # empty, else 0.0
            return 1.0 if len(result_ids) == 0 else 0.0
        predicted_positives = len(result_ids)
        if predicted_positives == 0:
            return 0.0
        true_positives = len(result_ids & ground_truth_ids)
        return true_positives / predicted_positives

    def compute_recall(
        ground_truth: pd.DataFrame,
        query_result: pd.DataFrame,
        id_column: str = "id",
    ):
        """
        Computes the recall of the query result towards the ground truth.
        Assumes that both dataframes have an "id" column the uniquely identifies
        a row. The recall is computed based on the IDs in the result and the
        ground truth.
        """
        ground_truth_ids = (
            set(ground_truth[id_column]) if not ground_truth.empty else set()
        )
        result_ids = (
            set(query_result[id_column])
            if query_result is not None and not query_result.empty
            else set()
        )
        if len(ground_truth_ids) == 0:
            # If ground truth is empty, recall is 1.0 if result is also empty,
            # else 0.0
            return 1.0 if len(result_ids) == 0 else 0.0
        true_positives = len(result_ids & ground_truth_ids)
        actual_positives = len(ground_truth_ids)
        return true_positives / actual_positives

    def compute_f1_score(
        ground_truth: pd.DataFrame,
        query_result: pd.DataFrame,
        id_column: str = "id",
    ):
        """
        Computes the F1 score of the query result towards the ground truth.
        Assumes that both dataframes have an "id" column the uniquely identifies
        a row. The F1 score is computed based on the IDs in the result and the
        ground truth.
        """
        precision = GenericEvaluator.compute_precision(
            ground_truth, query_result, id_column=id_column
        )
        recall = GenericEvaluator.compute_recall(
            ground_truth, query_result, id_column=id_column
        )
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def compute_f1_score_classify(
        ground_truth: pd.DataFrame,
        query_result: pd.DataFrame,
        result_column: str,
        id_column: str = "id",
    ):
        """
        Computes the F1 score of the query result towards the ground truth.
        Assumes that both dataframes have an "id" column the uniquely identifies
        a row. The F1 score is computed based on the IDs in the result and the
        ground truth.
        """
        if ground_truth.shape[0] != query_result.shape[0]:
            raise ValueError(
                "Invalid results. Ground truth and query vectors should be of the same length."
            )

        gt = ground_truth.sort_values(id_column)[result_column]
        query = query_result.sort_values(id_column)[result_column]

        return f1_score(gt, query, average="macro")

    def compute_adjusted_rand_index(
        ground_truth: pd.DataFrame, query_result: pd.DataFrame
    ):
        """
        Computes the Adjusted Rand Index (ARI) between the group assignments in
        the query result and the ground truth. Assumes both dataframes have "id"
        and "group" columns.
        """
        # Handle None inputs gracefully
        if query_result is None:
            return 0.0

        # Map IDs to group assignments for both ground truth and query result
        gt_groups = ground_truth.set_index("id")["category"]
        qr_groups = query_result.set_index("id")["category"]

        common_ids = set(gt_groups.index) & set(qr_groups.index)
        if not common_ids:
            return 0.0

        gt_labels = [gt_groups[id] for id in common_ids]
        qr_labels = [qr_groups[id] for id in common_ids]
        return adjusted_rand_score(gt_labels, qr_labels)

    def compute_omega_index(ground_truth: pd.DataFrame, query_result: pd.DataFrame):
        # Convert to cluster lists. Consider items without category as a new group.
        ground_truth["category"] = ground_truth["category"].fillna(-1)
        query_result["category"] = query_result["category"].fillna(-1)

        gt_clusters = ground_truth.groupby("category")["id"].apply(list).tolist()
        pred_clusters = query_result.groupby("category")["id"].apply(list).tolist()

        gt_nc = NodeClustering(communities=gt_clusters, graph=None)
        pred_nc = NodeClustering(communities=pred_clusters, graph=None)

        return evaluation.omega(pred_nc, gt_nc).score

    def compute_accuracy_score(
        accuracy_metric_type: str,
        ground_truth: pd.DataFrame,
        query_result: pd.DataFrame,
        id_column: str = "id",
    ) -> SingleAccuracyScore:
        # Compute additional helper metrics if we have f1-score, precision, or recall
        if accuracy_metric_type in ["f1-score", "precision", "recall"]:
            f1_score = GenericEvaluator.compute_f1_score(
                ground_truth, query_result, id_column=id_column
            )
            precision = GenericEvaluator.compute_precision(
                ground_truth, query_result, id_column=id_column
            )
            recall = GenericEvaluator.compute_recall(
                ground_truth, query_result, id_column=id_column
            )

        if accuracy_metric_type == "f1-score":
            return SingleAccuracyScoreWithRetrievalDetails(
                f1_score,
                metric_type="f1-score",
                precision=precision,
                recall=recall,
                f1_score=f1_score,
            )
        elif accuracy_metric_type == "precision":
            return SingleAccuracyScoreWithRetrievalDetails(
                precision,
                metric_type="precision",
                precision=precision,
                recall=recall,
                f1_score=f1_score,
            )
        elif accuracy_metric_type == "recall":
            return SingleAccuracyScoreWithRetrievalDetails(
                recall,
                metric_type="recall",
                precision=precision,
                recall=recall,
                f1_score=f1_score,
            )
        elif accuracy_metric_type == "adjusted-rand-index":
            return SingleAccuracyScore(
                accuracy=GenericEvaluator.compute_adjusted_rand_index(
                    ground_truth, query_result
                ),
                metric_type="adjusted-rand-index",
            )
        elif accuracy_metric_type == "omega-index":
            return SingleAccuracyScore(
                accuracy=GenericEvaluator.compute_omega_index(
                    ground_truth, query_result
                ),
                metric_type="omega-index",
            )
        else:
            raise ValueError(
                f"Unsupported accuracy metric type: {accuracy_metric_type}"
            )
