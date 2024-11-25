
====================================
The Treeval score
====================================

Treeval was originally designed to evaluate the results of language models on **structured data extraction tasks**. This category of task is commonly associated to data serialized in format following `JSON schemas <https://json-schema.org/overview/what-is-jsonschema>`_ which supports `a few predefined leaf types <https://json-schema.org/understanding-json-schema/reference/type>`_: string, number, integer, boolean, array and null.

In an effort to standardize the evaluation of structured data extraction task, we designed the Treeval score as a simple and unified way to evaluate structured data. It is meant to be used with any language models to measure their performances on benchmarks and datasets. The Treeval score maps each of the following types to the metrics:

* ``integer``: :py:class:`treeval.metrics.ExactMatch`. Even if a reference and prediction integers are relatively close, most structured data retrieval tasks are purely extractive, i.e. the actual integer to extract is in most cases present in the data. In other cases, Large Language Models (LLMs) already perform quite well in making deductions, computations and reasoning to retrieve the expected integer. Using a metric measuring the relative distance between two values is feasible, but would be not enough penalize the average scores when computed over common benchmarks. Penalizing if the predicted value is not the expected value allows the Treeval score to be less tolerant and keep larger room for model performances improvements over common benchmarks;
* ``number``: :py:class:`treeval.metrics.ExactMatch`. This metric choice is motivated by the same reasons than for the integer type;
* ``boolean``: :py:class:`treeval.metrics.BooleanAccuracy`, which is identical to ExactMatch but works with booleans;
* ``string``: :py:class:`treeval.metrics.Levenshtein` and :py:class:`treeval.metrics.BERTScore`. The Levenshtein distance (edit distance) measures the minimum number of deletions, additions and editions combined to perform on a prediction string until it becomes identical to a reference string. It is therefore a string similarity metric, which is represented as a normalized ratio in the Treeval score;

You can directly compute the Treeval score of a batch of pairs of references/predictions using the :py:func:`treeval.treeval_score` method.
