<p align="center">
  <img src="docs/resources/treeval_logo.svg" alt="Treeval Logo"/>
</p>

[![PyPI version fury.io](https://badge.fury.io/py/treeval.svg)](https://pypi.python.org/pypi/treeval/)
[![Python 3.8](https://img.shields.io/badge/python-≥3.8-blue.svg)](https://www.python.org/downloads/release/)
[![Documentation Status](https://readthedocs.org/projects/treeval/badge/?version=latest)](https://treeval.readthedocs.io/en/latest/?badge=latest)
[![GitHub CI](https://github.com/NuMindAI/treeval/actions/workflows/pytest.yml/badge.svg)](https://github.com/NuMindAI/treeval/actions/workflows/pytest.yml)
[![Codecov](https://img.shields.io/codecov/c/github/NuMindAI/treeval)](https://codecov.io/gh/NuMindAI/treeval)
[![GitHub license](https://img.shields.io/github/license/NuMindAI/treeval.svg)](https://github.com/NuMindAI/treeval/blob/main/LICENSE)
[![Code style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
<!--[![Downloads](https://static.pepy.tech/badge/treeval)](https://pepy.tech/project/treeval)-->

## Introduction

Treeval is a Python package providing an easy and flexible way to evaluate the matching of tree-based data, i.e. dictionaries. It is initially developed to evaluate the results of [structured data extraction tasks](https://numind.ai/blog/nuextract-a-foundation-model-for-structured-extraction>) from language models, but can be used with any type of leaf values and metrics.

Treeval can be installed from PyPi by running:

```bash
pip install treeval
```

**Documentation:** [treeval.readthedocs.io](https://treeval.readthedocs.io/en/latest/)

## Usage example

```Python
from treeval import treeval, create_tree_metrics
from treeval.metrics import SacreBLEU, Accuracy, Levenshtein, ExactMatch


schema = {
    "song_name": "string",  # node names/keys are dictionary keys, node types are dictionary values.
    "song_duration_in_seconds": "integer",
    "has_lyrics": "boolean",  # if a node value is anything other than a dictionary, it is a leaf.
    "information": {  # a node value can be a nested dictionary, i.e. a branch
        "time_signature": ["4/4", "4/2", "2/2"],  # one of the element within the list
        "tempo": "integer",
    },
    "instruments": ["string"],  # list of items of type "string"
}

reference = {
    "song_name": "Die With A Smile",
    "song_duration_in_seconds": 320,
    "has_lyrics": True,
    "information": {"time_signature": "4/4", "tempo": 120},
    "instruments": ["piano", "synthesizer", "drums"],
}
prediction = {
    "song_name": "Die With A Smile",
    "song_duration_in_seconds": 320,
    "information": {"time_signature": "4/4", "tempo": 128},
    "instruments": ["piano", "synthesizer", "drums", "trumpet"],
    "artist": "Daft Punk",  # invented field that shouldn't be there
}  # the "has_lyrics" field is missing too

# Define which metrics to run for which leaves
types_metrics = {  # for types
    "string": ["exact_match", "levenshtein"],
    "integer": ["exact_match"],
    "boolean": ["exact_match"],
    (): ["exact_match"],  # choice among list, here "time_signature"
}
leaves_metrics = {  # for specific leaves, on top of the ones above
    "information": {"time_signature": ["sacrebleu"]},
}
tree_metrics = create_tree_metrics(schema, leaves_metrics, types_metrics)
metrics = {  # initialize metrics modules
    metric.name: metric
    for metric in [
        Accuracy(),
        SacreBLEU(),
        Levenshtein(),
        ExactMatch(),
    ]
}

results = treeval([prediction], [reference], schema, metrics, tree_metrics)
```

The `results` will be a dictionary having the same structure as the `schema`, reporting metrics results for each leaf, along with precision/recall/F1 scores:

```Python
{
    "song_name": {
        "exact_match": {"exact_match": 1.0},
        "levenshtein": {"levenshtein": 0.0, "levenshtein_ratio": 1.0},
    },
    "song_duration_in_seconds": {"exact_match": {"exact_match": 1.0}},
    "has_lyrics": None,
    "information": {
        "time_signature": {
            "exact_match": {"exact_match": 1.0},
            "sacrebleu": {
                "score": 0.0,
                "counts": [3, 2, 1, 0],
                "totals": [3, 2, 1, 0],
                "precisions": [100.0, 100.0, 100.0, 0.0],
                "bp": 1.0,
                "sys_len": 3,
                "ref_len": 3,
            },
        },
        "tempo": {"exact_match": {"exact_match": 0.0}},
    },
    "instruments": {
        "exact_match": {"exact_match": 1.0},
        "levenshtein": {"levenshtein": 1.0},
    },
    "precision_node": 0.8571428571428571,
    "recall_node": 0.8571428571428571,
    "f1_node": 0.8571428571428571,
    "precision_leaf": 1.0,
    "recall_leaf": 1.0,
    "f1_leaf": 1.0,
    "treeval_score": 0.5142857142857142,
}
```

For ease of read and interpretation, they can easily be aggregated per metric average, or per node type combined with metric average, using respectively the `aggregate_results_per_metric` and `aggregate_results_per_leaf_type` arguments of the `treeval` method. The former will reduce the results to:

```Python
{
    "levenshtein": 1.0,
    "sacrebleu": 0.0,
    "exact_match": 0.8,
    "precision_node": 0.8571428571428571,
    "f1_node": 0.8571428571428571,
    "recall_node": 0.8571428571428571,
    "f1_leaf": 1.0,
    "precision_leaf": 1.0,
    "recall_leaf": 1.0,
    "treeval_score": 0.5142857142857142,
}
```
