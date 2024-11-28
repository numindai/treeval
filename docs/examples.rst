
===================================
Code examples
===================================

This pages show Treeval usage examples.

Creating a schema and tree metrics
----------------------------------

The ``schema`` is the description of the tree. It is a dictionary mapping node names (keys) to leaf types (string values). The schema is used by the :py:func:`treeval.treeval` method to efficiently parse trees and batch metrics computations.

..  code-block:: python

    schema = {
        "song_name": "string",  # node name/key are dictionary keys, node types are dictionary values.
        "artist_name": "string",  # if a node value is anything other than a dictionary, it is a leaf.
        "song_duration_in_seconds": "integer",
        "has_lyrics": "boolean",
        "information": {  # a node value can be a nested dictionary, i.e. a branch
            "tempo": "integer",
            "time_signature": ["4/4", "4/2", "2/2"],  # one of the element within the list
            "key_signature": "string",
        },
        "instruments": ["string"],  # list of items of type "string"
    }

A schema is used in combination with a ``tree_metrics``, which is dictionary with the same structure (node names/keys) as the schema where leaves values are sets of names of the metrics to compute scores on the associated leaves.

..  code-block:: python

    schema = {
        "n1": "integer",
        "n2": "string",
        "n3": {"n4": "string", "n5": "boolean"},
    }
    tree_metrics = {
        "n1": {"accuracy"},
        "n2": {"bleu", "levenshtein"},
        "n3": {"n4": {"bleu"}, "n5": {"boolean_accuracy"}},
    }

In some cases, you might be working with complex trees of several depths and multiple branches that might be time consuming to create a metrics tree from, and simply want to map leaves types to specific sets of metrics. The :py:func:`treeval.create_tree_metrics` allows to easily do that. It also allows to include metrics to compute for specific leaves. Notice that the "boolean" type of the ``n5`` leaf is missing from the ``types_metrics`` dictionary. Trees following the schema can still be evaluated as its metrics are provided in the ``leaves_metrics`` dictionary.

..  code-block:: python

    from treeval import create_tree_metrics
    # mapping leaves types to lists of metrics to compute for them
    types_metrics = {
        "string": ["bleu"],
        "integer": ["accuracy"],
    }
    # metrics to compute for specific leaves
    leaves_metrics = {
        "n2": ["levenshtein"],
        "n3": {"n5": ["boolean_accuracy"]},
    }

    tree_metrics = create_tree_metrics(tree_schema, leaves_metrics, types_metrics)

Evaluating trees
-----------------------------

When you have a well defined schema and metrics tree, you can call the :py:func:`treeval.treeval` method to evaluate a batch of pairs of reference and hypothesis trees. Notice that the hypothesis trees might have mismatching branches, that will impact the precision/recall/f1 scores (:ref:`Precision, Recall, F1 and mismatching tree branches`). We also need to load the metrics modules before passing them to the method. You can read the ":ref:`Metrics in Treeval`" page to learn more about the metrics and how to create your own.

..  code-block:: python

    from treeval import treeval
    from treeval.metrics import BLEU, Accuracy, BooleanAccuracy, Levenshtein

    # Load the metrics modules, using their names as provided in ``tree_metrics``
    metrics = {
        metric.name: metric
        for metric in [
            Accuracy(),
            BLEU(),
            BooleanAccuracy(),
            Levenshtein(),
        ]
    }

    reference_trees = [
        {
            "n1": 10,
            "n2": "foo",
            "n3": {"n4": "foo", "n5": True},
        },
        {
            "n1": 20,
            "n2": "bar",
            "n3": {"n4": None, "n5": False},
        },
    ]
    hypothesis_trees = [
        {
            "n1": None,
            "n2": "foo",
            "n3": {"n4": "fooo", "n5": True},
        },
        {
            "n1": 20,
            "n2": "bar",
            "n3": {"n4": "barr"},
        },
    ]

    results = treeval(hypothesis_trees, reference_trees, schema, metrics, tree_metrics)
    print(results)

The above code block will print the "raw" results as a tree following the schema's structure:

.. code-block:: JSON

    {
        "n1": {"accuracy": {"accuracy": 1.0}},
        "n2": {
            "levenshtein": {"levenshtein": 0.0, "levenshtein_ratio": 1.0},
            "bleu": {
                "bleu": 0.0,
                "precisions": [1.0, 0.0, 0.0, 0.0],
                "brevity_penalty": 1.0,
                "length_ratio": 1.0,
                "translation_length": 2,
                "reference_length": 2,
            },
        },
        "n3": {
            "n4": {
                "bleu": {
                    "bleu": 0.0,
                    "precisions": [0.0, 0.0, 0.0, 0.0],
                    "brevity_penalty": 1.0,
                    "length_ratio": 1.0,
                    "translation_length": 1,
                    "reference_length": 1,
                }
            },
            "n5": {"boolean_accuracy": {"boolean_accuracy": 1.0}},
        },
        "precision_node": 1.0,
        "recall_node": 0.9,
        "f1_node": 0.9473684210526316,
        "precision_leaf": 0.8333333333333334,
        "recall_leaf": 0.8333333333333334,
        "f1_leaf": 0.8333333333333334,
    }

Aggregating results
-----------------------------

This complete results report provides great interpretability, but might be too complicated to analyze. The :py:func:`treeval.aggregate_results_per_metric` and :py:func:`treeval.aggregate_results_per_leaf_type` allow to aggregate them per metrics or node leaf type respectively. Calling the former by providing the results from the previous code block will reduce it to metrics scores averages:

.. code-block:: JSON

    {
        "accuracy": 1.0,
        "bleu": 0.0,
        "boolean_accuracy": 1.0,
        "f1_leaf": 0.8333333333333334,
        "f1_node": 0.9473684210526316,
        "levenshtein": 1.0,
        "precision_leaf": 0.8333333333333334,
        "precision_node": 1.0,
        "recall_leaf": 0.8333333333333334,
        "recall_node": 0.9,
    }

Calling :py:func:`treeval.aggregate_results_per_leaf_type` will provide the average metrics results per leaf type, providing a finer degree of details.

.. code-block:: JSON

    {
        "integer": {"accuracy": 1.0},
        "boolean": {"boolean_accuracy": 1.0},
        "string": {"bleu": 0.0, "levenshtein": 1.0},
        "recall_node": 0.9,
        "recall_leaf": 0.8333333333333334,
        "f1_leaf": 0.8333333333333334,
        "precision_node": 1.0,
        "f1_node": 0.9473684210526316,
        "precision_leaf": 0.8333333333333334,
    }
