"""Test validation methods."""

from __future__ import annotations

from pathlib import Path

from treeval.metrics import (
    BLEU,
    F1,
    MSE,
    Accuracy,
    BooleanAccuracy,
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
    BooleanAccuracy(),
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
