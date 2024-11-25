"""Treeval tree evaluation method."""

from __future__ import annotations

from metrics.metrics import BERTScore, BooleanAccuracy, ExactMatch, Levenshtein

METRICS = {BooleanAccuracy(), ExactMatch(), Levenshtein(), BERTScore()}
TYPES_METRICS = {
    "integer": {"exact_match"},
    "number": {"exact_match"},
    "boolean": {"boolean_accuracy"},
    "string": {"levenshtein", "bertscore"},
}
