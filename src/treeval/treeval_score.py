"""Treeval score setting."""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import bert_score
except ImportError:
    bert_score = None

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
) -> dict:
    """
    Treeval evaluation method.

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
            BERTScore() if bert_score else None,
        }
    }

    # TODO average of all leaves results, or metrics averages? +update docs when decided
    # Compute treeval
    return treeval(
        predictions,
        references,
        schema,
        metrics,
        tree_metrics,
        aggregate_results_per_metric=True,
        hierarchical_averaging=False,
    )
