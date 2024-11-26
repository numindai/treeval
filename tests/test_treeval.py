"""Tests for the treeval method."""

from __future__ import annotations

import pytest
from treeval import create_tree_metrics, treeval

from tests.utils_tests import COMPLETE_SCHEMA, METRICS

# Test cases
# [(schema, reference, prediction, tree_metrics, expected_scores)]
__simple_schema = {"n1": "string", "n2": "string", "n3": "integer"}
__simple_schema_tree_metrics = {"n1": {"sacrebleu"}, "n2": {"sacrebleu"}, "n3": {"mse"}}
__tree_metrics_complete_schema = create_tree_metrics(
    COMPLETE_SCHEMA,
    {
        "n1": ["sacrebleu", "exact_match"],
        "n3": ["boolean_accuracy"],
        "n5": ["accuracy"],
        "n7": ["sacrebleu"],
        "n8": {"n82": ["sacrebleu"]},
        "n10": {"n10_int": ["accuracy"], "n10_string": ["sacrebleu"]},
    },
    {
        "string_2": ["sacrebleu"],
        "integer": ["accuracy"],
        "number": ["accuracy"],
        "datetime": ["sacrebleu"],
        (): ["exact_match"],  # choice among list
    },
)
DATA_CASES = [
    (
        __simple_schema,
        [
            {"n1": "test1", "n2": "test12", "n3": 1},
            {"n1": "test2", "n2": "test22", "n3": 1},
            {"n1": "test3", "n2": "test32", "n3": 1},
            {"n1": "test4", "n2": "test42", "n3": 1},
        ],
        [
            {"n1": "test1", "n2": "test12", "n3": 1},
            {"n1": "test1", "n2": "test12", "n3": 1},
            {"n1": "test3", "n2": "anything", "n3": 1},
            {"foo": "bar", "n2": "anything", "n3": 0},
        ],
        __simple_schema_tree_metrics,
        {
            "n1": {
                "sacrebleu": {
                    "score": 0.0,
                    "counts": [2, 0, 0, 0],
                    "totals": [3, 0, 0, 0],
                    "precisions": [66.66666666666667, 0.0, 0.0, 0.0],
                    "bp": 1.0,
                    "sys_len": 3,
                    "ref_len": 3,
                }
            },
            "n2": {
                "sacrebleu": {
                    "score": 0.0,
                    "counts": [1, 0, 0, 0],
                    "totals": [4, 0, 0, 0],
                    "precisions": [25.0, 0.0, 0.0, 0.0],
                    "bp": 1.0,
                    "sys_len": 4,
                    "ref_len": 4,
                }
            },
            "n3": {"mse": {"mse": 0.25}},
            "precision_nodes": 0.9166666666666666,
            "recall_nodes": 0.9166666666666666,
            "f1_nodes": 0.9166666666666666,
        },
    ),
    (
        COMPLETE_SCHEMA,
        [
            {
                "n1": "Die With A Smile",
                "n2": "Lady Gaga, Bruno Mars",
                "n3": True,
                "n4": 1,
                "n5": 2.5,
                "n6": "2024-03-14",
                "n7": [
                    "I just woke up from a dream, where you and I had to say good-bye.",
                    "And I don't know what it all means, but since I survived I "
                    "realized.",
                    "Wherever you go, that's where I'll follow.",
                ],
                "n8": {"n81": 2, "n82": "test", "n83": "2023-07-10"},
                "n9": "low",
                "n10": [
                    {"n10_int": 10, "n10_string": "piano"},
                    {"n10_int": 9, "n10_string": "French horn"},
                    {"n10_int": 8, "n10_string": "drums"},
                    {"n10_int": 7, "n10_string": "drums"},
                ],
            }
        ],
        [
            {
                "n1": "Die With A Smile",
                "n2": "Lady Gaga, Bruno Mars",
                "n3": True,
                "n4": 1,
                "n5": 2.5,
                "n6": "2024-03-14",
                "n7": [
                    "I just woke up from a dream, where you and I had to say good-bye.",
                    "And I don't know what it all means, but since I survived I "
                    "realized.",
                    "Wherever you go, that's where I'll follow.",
                ],
                "n8": {"n81": 2, "n82": "test", "n83": "2023-07-10"},
                "n9": "low",
                "n10": [
                    {"n10_int": 10, "n10_string": "piano"},
                    {"n10_int": 9, "n10_string": "French born"},
                    {"n10_int": 6, "n10_string": "drums"},
                    {"n10_int": 5, "n10_string": "drums"},
                ],
            }
        ],
        __tree_metrics_complete_schema,
        {
            "n1": {
                "sacrebleu": {
                    "score": 100.00000000000004,
                    "counts": [4, 3, 2, 1],
                    "totals": [4, 3, 2, 1],
                    "precisions": [100.0, 100.0, 100.0, 100.0],
                    "bp": 1.0,
                    "sys_len": 4,
                    "ref_len": 4,
                },
                "exact_match": {"exact_match": 1.0},
            },
            "n2": {
                "sacrebleu": {
                    "score": 100.00000000000004,
                    "counts": [5, 4, 3, 2],
                    "totals": [5, 4, 3, 2],
                    "precisions": [100.0, 100.0, 100.0, 100.0],
                    "bp": 1.0,
                    "sys_len": 5,
                    "ref_len": 5,
                }
            },
            "n3": {"boolean_accuracy": {"boolean_accuracy": 1.0}},
            "n4": {"accuracy": {"accuracy": 1.0}},
            "n5": {"accuracy": {"accuracy": 1.0}},
            "n6": {
                "sacrebleu": {
                    "score": 100.00000000000004,
                    "counts": [5, 4, 3, 2],
                    "totals": [5, 4, 3, 2],
                    "precisions": [100.0, 100.0, 100.0, 100.0],
                    "bp": 1.0,
                    "sys_len": 5,
                    "ref_len": 5,
                }
            },
            "n7": {"sacrebleu": {"sacrebleu": 100.00000000000004}},
            "n8": {
                "n81": {"accuracy": {"accuracy": 1.0}},
                "n82": {
                    "sacrebleu": {
                        "score": 0.0,
                        "counts": [1, 0, 0, 0],
                        "totals": [1, 0, 0, 0],
                        "precisions": [100.0, 0.0, 0.0, 0.0],
                        "bp": 1.0,
                        "sys_len": 1,
                        "ref_len": 1,
                    }
                },
                "n83": {
                    "sacrebleu": {
                        "score": 100.00000000000004,
                        "counts": [5, 4, 3, 2],
                        "totals": [5, 4, 3, 2],
                        "precisions": [100.0, 100.0, 100.0, 100.0],
                        "bp": 1.0,
                        "sys_len": 5,
                        "ref_len": 5,
                    }
                },
            },
            "n9": {"exact_match": {"exact_match": 1.0}},
            "n10": {"accuracy": {"accuracy": 0.5}, "sacrebleu": {"sacrebleu": 0.0}},
            "precision_nodes": 1.0,
            "recall_nodes": 1.0,
            "f1_nodes": 1.0,
        },
    ),
]


# [(schema, references, predictions, tree_metrics, expected_scores)]
__prf_schema = {
    "n1": "integer",
    "n2": "integer",
    "n3": {"n4": "integer", "n5": "integer"},
}
__prf_tree_metrics = {
    "n1": {"mse"},
    "n2": {"mse"},
    "n3": {"n4": {"mse"}, "n5": {"mse"}},
}
PRF_CASES = [
    (
        __prf_schema,
        [{"n1": 1, "n2": 1, "n3": {"n4": 1, "n5": 1}}],
        [{"n1": 1, "n2": 1, "n3": {"n4": 1, "n5": 1}}],
        __prf_tree_metrics,
        (1, 1, 1),
    ),
    (
        __prf_schema,
        [{"n1": 1, "n2": 1, "n3": {"n4": 1, "n5": 1}}],
        [{"n1": 1, "n2": 1, "n3": {"n4": 1}}],
        __prf_tree_metrics,
        (1, 0.8, 0.888888888888889),
    ),
    (
        __prf_schema,
        [{"n1": 1, "n2": 1, "n3": {"n4": 1, "n5": 1}}],
        [{"n1": 1, "n2": 1, "n3": {"n4": 1, "n5": 1, "n6": 1}}],
        __prf_tree_metrics,
        (0.8333333333333334, 1, 0.9090909090909091),
    ),
    (
        __prf_schema,
        [{"n1": 1, "n2": 1, "n3": {"n4": 1, "n5": 1}}],
        [{"n1": 1, "n2": 1, "n3": {"n4": 1, "n6": 1}}],
        __prf_tree_metrics,
        (0.8, 0.8, 0.8),
    ),
]
__batched_prf_cases = [[], []]
for prf_case in PRF_CASES:
    __batched_prf_cases[0].append(prf_case[1][0])
    __batched_prf_cases[1].append(prf_case[2][0])
PRF_CASES.append(
    (
        __prf_schema,
        __batched_prf_cases[0],
        __batched_prf_cases[1],
        __prf_tree_metrics,
        (0.9, 0.9, 0.9),
    )
)


@pytest.mark.parametrize("data", DATA_CASES)
def test_treeval(data: tuple) -> None:
    """
    Test the :py:func:`treeval.treeval` method.

    :param data: data for the test case, including schema, references, predictions,
        metrics and expected results.
    """
    schema, references, predictions, tree_metrics, expected_scores = data
    results = treeval(predictions, references, schema, METRICS, tree_metrics)
    assert results == expected_scores


@pytest.mark.parametrize("data", PRF_CASES)
def test_treeval_precision_recall_f1(data: tuple) -> None:
    """
    Test the precision/recall/f1 of the :py:func:`treeval.treeval` method.

    :param data: data for the test case, including schema, references, predictions,
        metrics and expected results.
    """
    schema, reference, prediction, tree_metrics, expected_scores = data
    results = treeval(prediction, reference, schema, METRICS, tree_metrics)
    assert results["precision_nodes"] == pytest.approx(expected_scores[0])
    assert results["recall_nodes"] == pytest.approx(expected_scores[1])
    assert results["f1_nodes"] == pytest.approx(expected_scores[2])
