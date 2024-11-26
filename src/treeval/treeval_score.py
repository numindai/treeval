"""Treeval score setting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .metrics import BERTScore, BooleanAccuracy, ExactMatch, Levenshtein
from .treeval import create_tree_metrics, treeval

if TYPE_CHECKING:
    from collections.abc import Sequence


TYPES_METRICS = {
    "integer": ["exact_match"],
    "number": ["exact_match"],
    "boolean": ["boolean_accuracy"],
    "string": ["levenshtein", "bertscore"],
    (): ["exact_match"],
}


def treeval_score(
    predictions: Sequence[dict],
    references: Sequence[dict],
    schema: dict,
) -> float:
    """
    Treeval evaluation method.

    This method is equivalent to calling :py:func:`treeval.treeval` with a tree metrics
    built from the Treeval score types metrics, aggregating the results per metric,
    normalizing the scores and averaging them.

    :param predictions: list of dictionary predictions.
    :param references: list of dictionary references.
    :param schema: schema of the tree as a dictionary specifying each leaf type. The
        references must all follow this exact tree structure, while the predictions can
        have mismatching branches which will impact the tree precision/recall/f1 scores
        returned by the method.
    :return: the metrics results. The returned dictionary will have the same tree
        structure as the provided ``schema`` if ``aggregate_results_per_metric`` or
        ``aggregate_results_per_leaf_type`` are disabled, otherwise it will map metrics
        and/or leaf types to average results over all leaves.
    """
    # Create tree metrics
    tree_metrics = create_tree_metrics(schema, types_metrics=TYPES_METRICS)

    # metrics are initialized here, as they might require external dependencies that
    # shouldn't be required to run the rest of the library. If required dependencies are
    # missing, exceptions will be raised when loading the metrics.
    metrics = {
        m.name: m
        for m in {
            BooleanAccuracy(),
            ExactMatch(),
            Levenshtein(),
            BERTScore(),
        }
    }

    # Compute treeval
    results = treeval(
        predictions,
        references,
        schema,
        metrics,
        tree_metrics,
        aggregate_results_per_metric=True,
        hierarchical_averaging=False,
    )

    # Normalize scores and return average
    for metric_name, metric_score in results.copy().items():
        if metrics[metric_name].score_range != (0, 1):
            low_bound, high_bound = metrics[metric_name].score_range
            results[metric_name] = min(
                max(metric_score, low_bound), high_bound
            ) - low_bound / (high_bound - low_bound)
        if not metrics[metric_name].higher_is_better:
            results[metric_name] = 1 - results[metric_name]
    return sum(results.values()) / len(results)
