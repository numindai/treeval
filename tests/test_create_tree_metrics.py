"""Tests for the treeval method."""

from __future__ import annotations

import pytest
from treeval import create_tree_metrics

from tests.utils_tests import COMPLETE_SCHEMA

# Test cases
# (schema, leaves_metrics, types_metrics, tree_non_exclusive, tree_exclusive)
TEST_CASES = [
    (
        COMPLETE_SCHEMA,
        {
            "n1": ["sacrebleu", "exact_match"],
            "n3": ["sacrebleu", "exact_match"],
            "n5": ["accuracy", "f1"],
            "n7": ["sacrebleu", "f1"],
            "n8": {"n82": ["sacrebleu"]},
            "n9": ["sacrebleu"],
            "n10": {"n10_int": ["accuracy"], "n10_string": ["sacrebleu"]},
        },
        {
            "string_2": ["sacrebleu"],
            "integer": ["accuracy"],
            "number": ["exact_match"],
            "datetime": ["sacrebleu"],
        },
        {
            "n1": {"exact_match", "sacrebleu"},
            "n2": {"sacrebleu"},
            "n3": {"exact_match", "sacrebleu"},
            "n4": {"accuracy"},
            "n5": {"accuracy", "f1", "exact_match"},
            "n6": {"sacrebleu"},
            "n7": {"sacrebleu", "f1"},
            "n8": {"n81": {"accuracy"}, "n82": {"sacrebleu"}, "n83": {"sacrebleu"}},
            "n9": {"sacrebleu"},
            "n10": {"n10_int": {"accuracy"}, "n10_string": {"sacrebleu"}},
        },
        {
            "n1": {"sacrebleu", "exact_match"},
            "n2": {"sacrebleu"},
            "n3": {"sacrebleu", "exact_match"},
            "n4": {"accuracy"},
            "n5": {"accuracy", "f1"},
            "n6": {"sacrebleu"},
            "n7": {"sacrebleu", "f1"},
            "n8": {"n81": {"accuracy"}, "n82": {"sacrebleu"}, "n83": {"sacrebleu"}},
            "n9": {"sacrebleu"},
            "n10": {"n10_int": {"accuracy"}, "n10_string": {"sacrebleu"}},
        },
    ),
    (
        COMPLETE_SCHEMA,
        {
            "n1": ["sacrebleu", "exact_match"],
            "n3": ["sacrebleu", "exact_match"],
            "n5": ["accuracy", "f1"],
            "n8": {"n82": ["sacrebleu"]},
        },
        {
            "integer": ["accuracy"],
            "number": ["exact_match"],
            "datetime": ["sacrebleu"],
        },
        None,
        None,
    ),
]


@pytest.mark.parametrize("schema_metrics", TEST_CASES)
@pytest.mark.parametrize("exclusive_leaves_types_metrics", [False, True])
def test_create_tree_metrics(
    schema_metrics: tuple, exclusive_leaves_types_metrics: bool
) -> None:
    """
    Test that all the leaves from the data samples can be evaluated.

    :param schema_metrics: schema and leaves/types metrics to build the
        ``metrics tree`` from.
    :param exclusive_leaves_types_metrics:
    """
    schema, leaves_metrics, types_metrics, tree_non_exclusive, tree_exclusive = (
        schema_metrics
    )

    # Non-passing cases, should catch the error raised
    if (exclusive_leaves_types_metrics and tree_exclusive is None) or (
        not exclusive_leaves_types_metrics and tree_non_exclusive is None
    ):
        try:
            _ = create_tree_metrics(
                schema, leaves_metrics, types_metrics, exclusive_leaves_types_metrics
            )
            pytest.fail("Should have fail")
        except ValueError:
            pass  # ok, should fail

    # Passing case
    else:
        tree_metrics = create_tree_metrics(
            schema, leaves_metrics, types_metrics, exclusive_leaves_types_metrics
        )
        assert (
            tree_metrics == tree_exclusive
            if exclusive_leaves_types_metrics
            else tree_non_exclusive
        )
