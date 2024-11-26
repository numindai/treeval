"""Tests for the treeval method."""

from __future__ import annotations

import pytest
from treeval import aggregate_results_per_leaf_type, aggregate_results_per_metric
from treeval.treeval import _PRF_METRIC_NAMES

from tests.utils_tests import METRICS

# Treeval results
# (schema, tree_results, aggregated_results_metrics, aggregated_results_leaf_type)
__schema = {
    "n1": "integer",
    "n2": "integer",
    "n3": {"n4": "integer", "n5": "string"},
}
RESULTS_CASES = [
    (
        __schema,
        {
            "n1": {"accuracy": {"score": 1}, "mse": {"mse": 0}},
            "n2": {"accuracy": {"score": 1}, "mse": {"mse": 0}},
            "n3": {
                "n4": {"accuracy": {"score": 1}, "mse": {"mse": 0}},
                "n5": {"bleu": {"bleu": 1}, "mse": {"mse": 0}},
            },
            "precision_nodes": 1,
            "recall_nodes": 1,
            "f1_nodes": 1,
        },
        {"accuracy": 1, "mse": 0, "bleu": 1},
        {"integer": {"accuracy": 1, "mse": 0}, "string": {"bleu": 1, "mse": 0}},
    ),
    (
        __schema,
        {
            "n1": {"accuracy": {"score": 0}, "mse": {"mse": 0}},
            "n2": {"accuracy": {"score": 0.5}, "mse": {"mse": 0}},
            "n3": {
                "n4": {"accuracy": {"score": 1}, "mse": {"mse": 0}},
                "n5": {"bleu": {"bleu": 1}, "mse": {"mse": 1}},
            },
            "precision_nodes": 1,
            "recall_nodes": 1,
            "f1_nodes": 1,
        },
        {"accuracy": 0.5, "mse": 0.25, "bleu": 1},
        {"integer": {"accuracy": 0.5, "mse": 0}, "string": {"bleu": 1, "mse": 1}},
    ),
]


def get_tree_metrics_from_tree_results(tree_results: dict, schema: dict) -> dict:
    tree_metrics = {}
    for key, value in schema.items():
        if isinstance(value, dict):
            tree_metrics[key] = get_tree_metrics_from_tree_results(
                tree_results[key], value
            )
        else:
            tree_metrics[key] = set(tree_results[key])
    return tree_metrics


@pytest.mark.parametrize("data", RESULTS_CASES)
def test_aggregate_results(data: tuple) -> None:
    schema, tree_results, results_metrics, results_leaf_type = data
    tree_metrics = get_tree_metrics_from_tree_results(tree_results, schema)

    # Check metrics aggregation
    metrics_results = aggregate_results_per_metric(tree_results, tree_metrics, METRICS)
    for prf_key in _PRF_METRIC_NAMES:
        del metrics_results[prf_key]
    assert metrics_results == results_metrics

    # Check leaf type aggregation
    leaf_types_results = aggregate_results_per_leaf_type(tree_results, schema, METRICS)
    for prf_key in _PRF_METRIC_NAMES:
        del leaf_types_results[prf_key]
    assert leaf_types_results == results_leaf_type
