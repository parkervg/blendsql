"""
Created on July 27, 2025

@author: Jiale Lao

MovieEvaluator Implementation based on generic_evaluator
Uses DuckDB with ground truth SQL queries to generate reference results
"""

import pandas as pd

from .generic_evaluator import (
    GenericEvaluator,
    QueryMetricRetrieval,
    QueryMetricAggregation,
    QueryMetricRank,
)


class MovieEvaluator(GenericEvaluator):
    """Evaluator for the movie benchmark using the reusable framework."""

    def evaluate_single_query(
        self, query_id: int, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> "QueryMetricRetrieval | QueryMetricAggregation | QueryMetricRank":
        """Evaluate a single query based on its type."""
        evaluate_fn = self._discover_evaluate_impl(query_id)
        return evaluate_fn(system_results, ground_truth)

    def _evaluate_q1(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricRetrieval:
        """Q1: Five clearly positive reviews (any movie) - retrieval query with limit."""
        return self._generic_retrieval_limit_evaluation(
            system_results, ground_truth, limit=5
        )

    def _evaluate_q2(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricRetrieval:
        """Q2: Five positive reviews for movie "taken_3" - retrieval query with limit."""
        return self._generic_retrieval_limit_evaluation(
            system_results, ground_truth, limit=5
        )

    def _evaluate_q3(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricAggregation:
        """Q3: Count of positive reviews for movie "taken_3" - aggregation query."""
        return self._generic_aggregation_evaluation(system_results, ground_truth)

    def _evaluate_q4(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricAggregation:
        """Q4: Positivity ratio of reviews for movie "taken_3" - aggregation query."""
        return self._generic_aggregation_evaluation(system_results, ground_truth)

    def _evaluate_q5(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricRetrieval:
        """Q5: Pairs of reviews that express the same sentiment for movie with id '1189217-angels_and_demons' - retrieval query with limit."""
        return self._evaluate_review_pairs_with_limit(
            system_results, ground_truth, limit=10
        )

    def _evaluate_q6(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricRetrieval:
        """Q6: Pairs of reviews that express the opposite sentiment for movie with id '1189217-angels_and_demons' - retrieval query with limit."""
        return self._evaluate_review_pairs_with_limit(
            system_results, ground_truth, limit=10
        )

    def _evaluate_q7(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricRetrieval:
        """Q7: All Pairs of reviews that express the *opposite* sentiment for movie with id 'ant_man_and_the_wasp_quantumania' - retrieval query without limit."""
        return self._evaluate_review_pairs(system_results, ground_truth)

    def _evaluate_q8(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricAggregation:
        """Q8: Calculate the number of positive and negative reviews for movie "taken_3" - aggregation query."""
        return self._evaluate_sentiment_counts(system_results, ground_truth)

    def _evaluate_q9(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricRank:
        """Q9: Score from 1 to 5 how much did the reviewer like the movie - ranking query."""
        return self._generic_ranking_evaluation(system_results, ground_truth)

    def _evaluate_q10(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricRank:
        """Q10: Rank the movies based on movie reviews - ranking query."""
        return self._generic_ranking_evaluation(system_results, ground_truth)

    def _evaluate_q11(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricRank:
        return self._generic_retrieval_limit_evaluation(
            system_results, ground_truth, limit=100000
        )

    def _evaluate_q12(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricRank:
        return self._generic_retrieval_limit_evaluation(
            system_results, ground_truth, limit=2
        )

    def _generic_retrieval_limit_evaluation(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame, limit: int = 5
    ) -> QueryMetricRetrieval:
        """
        Generic evaluation for retrieval queries WITH limit clauses.

        For queries with LIMIT, we evaluate based on whether the returned items are valid
        (exist in ground truth) rather than exact set matching.
        Uses first column for comparison regardless of column names.
        """
        # BEFORE DOING ANYTHING - apply limit
        system_results = system_results.head(limit)

        if len(system_results) == 0:
            return QueryMetricRetrieval(
                precision=1.0 if len(ground_truth) == 0 else 0.0
            )
        if len(ground_truth) == 0:
            return QueryMetricRetrieval()

        # Use first column for comparison regardless of column names
        if len(system_results.columns) > 0 and len(ground_truth.columns) > 0:
            sys_col = system_results.columns[0]
            gt_col = ground_truth.columns[0]

            sys_ids = set(system_results[sys_col].dropna())
            gt_ids = set(ground_truth[gt_col].dropna())
            valid_results = sys_ids & gt_ids

            precision = len(valid_results) / len(sys_ids) if sys_ids else 0.0
            # For limit queries, recall is measured against the limit, not the full ground truth
            recall = (
                (len(valid_results) if len(valid_results) <= limit else limit)
                / min(limit, len(gt_ids))
                if gt_ids
                else 0.0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0.0
            )

            return QueryMetricRetrieval(precision, recall, f1)
        else:
            # Fallback to generic retrieval evaluation
            return self._generic_retrieval_evaluation(system_results, ground_truth)

    def _evaluate_review_pairs(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricRetrieval:
        """
        Evaluate queries that return pairs of reviews (Q5, Q6).

        Uses column positions instead of names: assumes first 3 columns are id, reviewId1, reviewId2.
        """
        if len(system_results) == 0:
            return QueryMetricRetrieval(
                precision=1.0 if len(ground_truth) == 0 else 0.0
            )
        if len(ground_truth) == 0:
            return QueryMetricRetrieval()

        # Ensure we have at least 3 columns (id, reviewId1, reviewId2)
        if len(system_results.columns) < 3 or len(ground_truth.columns) < 3:
            return QueryMetricRetrieval()

        def create_pair_tuple(row, columns):
            """Create a normalized tuple for pair comparison using column positions."""
            if len(columns) >= 3:
                movie_id = row[columns[0]]  # First column: movie id
                val1 = row[columns[1]]  # Second column: reviewId1
                val2 = row[columns[2]]  # Third column: reviewId2

                if pd.notna(val1) and pd.notna(val2) and pd.notna(movie_id):
                    base_tuple = tuple(sorted([val1, val2]))
                    return (movie_id, base_tuple)
            return None

        # Create sets of normalized pairs using column positions
        sys_cols = list(system_results.columns)
        gt_cols = list(ground_truth.columns)

        sys_pairs = {
            t
            for t in system_results.apply(
                lambda r: create_pair_tuple(r, sys_cols), axis=1
            )
            if t is not None
        }
        gt_pairs = {
            t
            for t in ground_truth.apply(lambda r: create_pair_tuple(r, gt_cols), axis=1)
            if t is not None
        }

        # Calculate metrics
        correct = sys_pairs & gt_pairs
        precision = len(correct) / len(sys_pairs) if sys_pairs else 0.0
        recall = len(correct) / len(gt_pairs) if gt_pairs else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        return QueryMetricRetrieval(precision, recall, f1)

    def _evaluate_review_pairs_with_limit(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame, limit: int = 10
    ) -> QueryMetricRetrieval:
        """
        Evaluate queries that return pairs of reviews (Q5, Q6) with limit-aware recall calculation.

        For queries with implicit limits (like "Ten Pairs"), recall is calculated against the limit,
        not the total ground truth pairs, to fairly evaluate systems that return the expected number.

        Uses column positions instead of names: assumes first 3 columns are id, reviewId1, reviewId2.
        """
        # BEFORE DOING ANYTHING - apply limit
        system_results = system_results.head(limit)

        if len(system_results) == 0:
            return QueryMetricRetrieval(
                precision=1.0 if len(ground_truth) == 0 else 0.0
            )
        if len(ground_truth) == 0:
            return QueryMetricRetrieval()

        # Ensure we have at least 3 columns (id, reviewId1, reviewId2)
        if len(system_results.columns) < 3 or len(ground_truth.columns) < 3:
            return QueryMetricRetrieval()

        def create_pair_tuple(row, columns):
            """Create a normalized tuple for pair comparison using column positions."""
            if len(columns) >= 3:
                movie_id = row[columns[0]]  # First column: movie id
                val1 = row[columns[1]]  # Second column: reviewId1
                val2 = row[columns[2]]  # Third column: reviewId2

                if pd.notna(val1) and pd.notna(val2) and pd.notna(movie_id):
                    base_tuple = tuple(sorted([val1, val2]))
                    return (movie_id, base_tuple)
            return None

        # Create sets of normalized pairs using column positions
        sys_cols = list(system_results.columns)
        gt_cols = list(ground_truth.columns)

        sys_pairs = {
            t
            for t in system_results.apply(
                lambda r: create_pair_tuple(r, sys_cols), axis=1
            )
            if t is not None
        }
        gt_pairs = {
            t
            for t in ground_truth.apply(lambda r: create_pair_tuple(r, gt_cols), axis=1)
            if t is not None
        }

        # Calculate metrics with limit-aware recall
        correct = sys_pairs & gt_pairs

        precision = len(correct) / len(sys_pairs) if sys_pairs else 0.0
        # For limit queries, recall is measured against the limit, not the full ground truth
        recall = (
            (len(correct) if len(correct) <= limit else limit)
            / min(limit, len(gt_pairs))
            if gt_pairs
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        return QueryMetricRetrieval(precision, recall, f1)

    def _evaluate_sentiment_counts(
        self, system_results: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> QueryMetricAggregation:
        """
        Evaluate Q7 sentiment count query (GROUP BY scoreSentiment with COUNT).

        Compares counts for each sentiment type and calculates aggregation errors.
        """
        if len(system_results) == 0 or len(ground_truth) == 0:
            return QueryMetricAggregation(
                relative_error=1.0,
                absolute_error=float("inf"),
                mean_absolute_percentage_error=100.0,
            )

        # Use column positions: first column is sentiment, second is count
        if len(system_results.columns) < 2 or len(ground_truth.columns) < 2:
            return QueryMetricAggregation(
                relative_error=1.0,
                absolute_error=float("inf"),
                mean_absolute_percentage_error=100.0,
            )

        sys_sentiment_col = system_results.columns[0]
        sys_count_col = system_results.columns[1]
        gt_sentiment_col = ground_truth.columns[0]
        gt_count_col = ground_truth.columns[1]

        # Create dictionaries for easy lookup
        sys_counts = {}
        for _, row in system_results.iterrows():
            sentiment = row[sys_sentiment_col]
            count = row[sys_count_col]
            if pd.notna(sentiment) and pd.notna(count):
                try:
                    sys_counts[str(sentiment).strip().upper()] = float(count)
                except (ValueError, TypeError):
                    continue

        gt_counts = {}
        for _, row in ground_truth.iterrows():
            sentiment = row[gt_sentiment_col]
            count = row[gt_count_col]
            if pd.notna(sentiment) and pd.notna(count):
                try:
                    gt_counts[str(sentiment).strip().upper()] = float(count)
                except (ValueError, TypeError):
                    continue

        if not sys_counts or not gt_counts:
            return QueryMetricAggregation(
                relative_error=1.0,
                absolute_error=float("inf"),
                mean_absolute_percentage_error=100.0,
            )

        # Calculate errors for each sentiment type
        total_absolute_error = 0.0
        total_relative_error = 0.0
        valid_comparisons = 0

        # Get all sentiment types from both results
        all_sentiments = set(sys_counts.keys()) | set(gt_counts.keys())

        for sentiment in all_sentiments:
            sys_count = sys_counts.get(sentiment, 0.0)
            gt_count = gt_counts.get(sentiment, 0.0)

            # Calculate absolute error
            abs_error = abs(sys_count - gt_count)
            total_absolute_error += abs_error

            # Calculate relative error (avoid division by zero)
            if gt_count != 0:
                rel_error = abs_error / abs(gt_count)
                total_relative_error += rel_error
                valid_comparisons += 1
            elif sys_count != 0:
                # Ground truth is 0 but system predicted non-zero
                total_relative_error += 1.0
                valid_comparisons += 1
            # If both are 0, no error to add

        # Calculate final metrics
        result = QueryMetricAggregation()
        result.absolute_error = total_absolute_error

        if valid_comparisons > 0:
            result.relative_error = total_relative_error / valid_comparisons
            result.mean_absolute_percentage_error = result.relative_error * 100
        else:
            result.relative_error = 0.0
            result.mean_absolute_percentage_error = 0.0

        return result
