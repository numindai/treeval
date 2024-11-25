"""
Metrics in Treeval: how to use your own and the ones implemented in Treeval.

Definition of metrics
---------------------

Any metric can be used as long as it is a callable object (or implements a
``.compute()`` method) taking two keyword arguments ``predictions`` and ``references``.
When evaluating lists of items (:ref:`Evaluation of lists of items`), Treeval will
normalize metrics scores and thus require two additional information: its score value
range and its "direction", e.g. ``(0, 1)`` and positive ("higher is better")
respectively for :py:class:`treeval.metrics.BLEU`.

The :py:class:`treeval.metrics.TreevalMetric` class allows you to create a wrapper
over any metric and provide its score value range and direction, so that it can be used
in Treeval. It is designed to work flawlessly with ``evaluate.EvaluationModule`` metrics
loaded from the Hugging Face evaluate library.

Treeval also implements wrappers for popular metrics.

Metrics wrappers
-----------------

Wrappers for popular metrics to be used out-of-the-box with Treeval.
"""

from .metrics import (
    BLEU,
    F1,
    MSE,
    ROUGE,
    Accuracy,
    BooleanAccuracy,
    ExactMatch,
    Levenshtein,
    Precision,
    Recall,
    SacreBLEU,
    TreevalMetric,
)

__all__ = [
    "BLEU",
    "F1",
    "MSE",
    "ROUGE",
    "Accuracy",
    "BooleanAccuracy",
    "ExactMatch",
    "Precision",
    "Recall",
    "SacreBLEU",
    "TreevalMetric",
    "Levenshtein",
]
