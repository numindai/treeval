"""Test validation methods."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
from treeval.metrics import (
    BLEU,
    F1,
    MSE,
    Accuracy,
    ExactMatch,
    SacreBLEU,
)

SEED = 777

HERE = Path(__file__).parent
TEST_LOG_DIR = HERE / "test_logs"


# Metrics
METRICS = [
    F1(),
    Accuracy(),
    ExactMatch(),
    BLEU(),
    SacreBLEU(),
    MSE(),
]
METRICS = {metric.name: metric for metric in METRICS}


# Complete schema covering different types and lists/dictionaries
COMPLETE_SCHEMA = {
    "n1": "string",
    "n2": "string_2",
    "n3": "bool",
    "n4": "integer",
    "n5": "number",
    "n6": "datetime",
    "n7": ["string"],
    "n8": {"n81": "integer", "n82": "string", "n83": "datetime"},
    "n9": ["low", "medium", "high"],
    "n10": [{"n10_int": "integer", "n10_string": "string"}],
    # TODO multilabel classification
}


def trees_approx_equal(
    reference: Mapping,
    hypothesis: Mapping,
    rel_tolerance: float | None = None,
    abs_tolerance: float | None = None,
    nan_ok: bool = False,
) -> bool:
    """
    Recursively check that two trees are approximately equal.

    :param reference: reference tree.
    :param hypothesis: hypothesis tree.
    :param rel_tolerance: relative tolerance.
    :param abs_tolerance: absolute tolerance.
    :param nan_ok: are "nan" ok.
    :return: whether the two trees are approximately equal.
    """
    for key, value in reference.items():
        if key not in hypothesis:
            return False
        value_hyp = hypothesis[key]
        if isinstance(value, Mapping):
            branch_approx = trees_approx_equal(
                value, value_hyp, rel_tolerance, abs_tolerance, nan_ok
            )
            if not branch_approx:
                return False
        elif value != pytest.approx(value, rel_tolerance, abs_tolerance, nan_ok):
            return False
    return True
