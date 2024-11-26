"""
Main treeval module exposing the main methods.

In a nutshell
**************

* :py:func:`treeval.treeval` is the treeval method computing metrics over a batch of
  pairs of references and predictions;
* :py:func:`treeval.create_tree_metrics` provides an easy way to create a
  ``tree_metrics``, to be used with py:func:`treeval.treeval` , from mappings of leaf
  types and metrics names;
* :py:func:`treeval.aggregate_results_per_metric` and
  :py:func:`treeval.aggregate_results_per_leaf_type` aggregates the tree results of
  :py:func:`treeval.treeval` per metric and/or leaf type;
* :py:func:`treeval.load_json_files` is a useful method loading a list of JSON files and
  decoding them into dictionaries;
* :py:func:`treeval.treeval_score` computes :ref:`The Treeval score` from the predefined
  metrics per type, useful to evaluate structured data extraction tasks.

Methods
-------
Detailed documentation:

"""

from .treeval import (
    _aggregate_results_per_leaf_type as aggregate_results_per_leaf_type,
)
from .treeval import (
    _aggregate_results_per_metric as aggregate_results_per_metric,
)
from .treeval import (
    create_tree_metrics,
    treeval,
)
from .treeval_score import treeval_score
from .utils import load_json_files

__all__ = [
    "aggregate_results_per_leaf_type",
    "aggregate_results_per_metric",
    "create_tree_metrics",
    "treeval",
    "treeval_score",
    "load_json_files",
]
