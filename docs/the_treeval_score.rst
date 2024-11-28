
====================================
The Treeval score
====================================

The :ref:`Precision, Recall, F1 and mismatching tree branches` introduces the way Treeval identify the leaves to compute metrics from, and how it computes precision, recall and f1 scores at the node and leaf level. These three scores are hierarchically related in how they are computed, yet can have opposite values, e.g. a pair of trees can have very different structures with few matching branches, resulting in a low node F1 score, yet identical matching leaves thus high metrics scores.

Yet, when evaluating results (generally whatever they may be), having a final unique score value is often very convenient for communication purposes. For this reason, we report an additional score called the **Treeval score** that takes into account the three aforementioned components into a single score. It is designed as a simple and unified way to evaluate tree similarity taking all aspects in consideration.

The Treeval score formula
-------------------------

The Treeval score is the product of the average of metrics scores, node F1 and leaf F1: :math:`\text{Treeval Score} = \bar{\mathcal{S}} \times F1_{node} \times F1_{leaf}`, where :math:`\mathcal{S}` is the set of metrics scores obtained with the :py:func:`treeval.aggregate_results_per_metric` method. Their product ensures that a low value for any of them will negatively impact the final score.

The Treeval score is automatically computed and reported by the :py:func:`treeval.treeval` method.

Default Treeval score metrics
-----------------------------

Treeval was originally designed to evaluate the results of language models on **structured data extraction tasks**. This category of task is commonly associated to data serialized in format following `JSON schemas <https://json-schema.org/overview/what-is-jsonschema>`_ which supports `a few predefined leaf types <https://json-schema.org/understanding-json-schema/reference/type>`_: string, number, integer, boolean, array and null.

In an effort to standardize the evaluation of structured data extraction task, Treeval proposes a set of specific metrics to be used for these common leaf types:

* ``integer``: :py:class:`treeval.metrics.ExactMatch`. Even if a reference and prediction integers are relatively close, most structured data retrieval tasks are purely extractive, i.e. the actual integer to extract is in most cases present in the data. In other cases, Large Language Models (LLMs) already perform quite well in making deductions, computations and reasoning to retrieve the expected integer. Using a metric measuring the relative distance between two values is feasible, but would be not enough penalize the average scores when computed over common benchmarks. Penalizing if the predicted value is not the expected value allows the Treeval score to be less tolerant and keep larger room for model performances improvements over common benchmarks;
* ``number``: :py:class:`treeval.metrics.ExactMatch`. This metric choice is motivated by the same reasons than for the integer type;
* ``boolean``: :py:class:`treeval.metrics.ExactMatch`, which acts as an accuracy score;
* ``string``: :py:class:`treeval.metrics.Levenshtein` and :py:class:`treeval.metrics.BERTScore`. The Levenshtein distance (edit distance) measures the minimum number of deletions, additions and editions combined to perform on a prediction string until it becomes identical to a reference string. It is therefore a string similarity metric, which is represented as a normalized ratio in the Treeval score;
* ``()`` (choice among list): :py:class:`treeval.metrics.ExactMatch`, which acts as an accuracy score.

To use this default set of metrics, you can use the :py:func:`treeval.create_treeval_score_default_tree_metrics` method to create a tree metrics tailored to your schema.

Example
-------

..  code-block:: python

    from treeval import treeval_score

    tree_schema = {
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
