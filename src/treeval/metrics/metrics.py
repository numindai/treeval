"""
Implements the metrics components.

In some cases, especially when dealing with lists of elements, treeval needs know how
the metrics behave in order to optimize the evaluation in the best possible way.
Namely, treeval requires to know the "direction" of the metric (higher/lower is better)
and the range of the values of its score in order to normalize them.
Instead of letting the user provide all these elements within tuples in a messy way, we
prefer to use a wrapper :py:class:`treeval.TreevalMetric` class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import evaluate
import numpy as np
from evaluate import EvaluationModule

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any


DEFAULT_SCORE_KEY = "score"


class TreevalMetric:
    """
    Treeval metric wrapper.

    The purpose of this class is to hold a metric module along with the range of value
    of its scores, and whether it should be maximized (higher is better) or not.

    :param module: a callable objects or an ``evaluate.EvaluationModule`` instance from
        the Hugging Face evaluate library.
    :param name: name of the metric. This name should be the same as provided in the
        ``tree_metrics`` argument of the :py:func:`treeval.treeval` method.
        If you provided an ``evaluate.EvaluationModule`` ``module``, this argument is
        optional and the name of the module will be used instead.
    :param score_range: range of the metric score values as a tuple. This information is
        required to normalize metric scores. (default: ``(0, 1)``)
    :param higher_is_better: direction of the metric. This information is required to
        normalize metric scores. (default: ``False``)
    :param kwargs: any keyword argument that should be passed to the ``module`` when
        called to compute metric scores.
    """

    def __init__(
        self,
        module: Callable | EvaluationModule | None,
        name: str | None = None,
        score_range: tuple[float | int, float | int] = (0, 1),
        higher_is_better: bool = True,
        **kwargs,
    ) -> None:
        self._module = module
        self._is_hf_module = isinstance(self._module, EvaluationModule)
        if name is None and self._is_hf_module:
            self.name = self._module.name
        else:
            if name is None:
                msg = "The `module` is not a `name`"
                raise ValueError(msg)
            self.name = name
        self.score_range = score_range
        self.higher_is_better = higher_is_better
        self._kwargs = kwargs

    def compute(
        self,
        predictions: Sequence[Any],
        references: Sequence[Any],
    ) -> dict:
        """
        Compute the metric score between pairs of references and predictions.

        This method does not take any keyword arguments. Keyword arguments can be passed
        to the module by providing them at the :py:class:`treeval.metrics.TreevalMetric`
        object initialization. This can also be achieved by subclassing this class and
        overriding this method.

        :param predictions: list of predictions to evaluate.
        :param references: expected reference values.
        :return: the score as a dictionary. The absolute score, which is the average of
            the score of all individual pairs of reference/prediction, should be the
            value of an entry with the key being either "score" or the name of the
            metric.
        """
        return self._compute(predictions, references)

    def _compute(
        self,
        predictions: Sequence[Any],
        references: Sequence[Any],
    ) -> dict:
        """
        Compute the metric score between pairs of references and predictions.

        This method is intended to be overridden by child classes if required.

        :param predictions: list of predictions to evaluate.
        :param references: expected reference values.
        :return: the score as a dictionary. The absolute score, which is the average of
            the score of all individual pairs of reference/prediction, should be the
            value of an entry with the key being either "score" or the name of the
            metric.
        """
        if self._is_hf_module:
            return self._module.compute(
                predictions=predictions, references=references, **self._kwargs
            )
        return self._module(
            predictions=predictions, references=references, **self._kwargs
        )

    def __call__(
        self,
        predictions: Sequence[Any],
        references: Sequence[Any],
    ) -> dict:
        """
        Compute the metric score between pairs of references and predictions.

        This method does not take any keyword arguments. If you need to provide
        additional arguments to the module, we recommend to modify the implementation of
        the module to handle this case.

        :param predictions: list of predictions to evaluate.
        :param references: expected reference values.
        :return: the score as a dictionary. The absolute score, which is the average of
            the score of all individual pairs of reference/prediction, should be the
            value of an entry with the key being either "score" or the name of the
            metric.
        """
        return self.compute(predictions, references)

    def get_metric_score(self, metric_result: dict) -> float:
        """
        Return the absolute score value of the results returned by the metric.

        By default, the metric modules returns a dictionary mapping specific results to
        their values. This allows to compute a metric, return its value along with
        additional results. This method returns the absolute metric score. By default,
        it is either mapped by the "score" key, or the key equal to the name of the
        metric (``metric.name``). This method is especially used when assigning pairs of
        predictions and references when evaluating lists of items.
        It can be overridden to handle specific cases.

        :param metric_result: metric results as a dictionary.
        :return: absolute metric score value as a floating point number.
        """
        return self._get_metric_score(metric_result)

    def _get_metric_score(self, metric_result: dict) -> float:
        """
        Return the absolute score value of the results returned by the metric.

        This method is intended to be overridden by child classes if required.

        :param metric_result: metric results as a dictionary.
        :return: absolute metric score value as a floating point number.
        """
        if self.name in metric_result:
            return metric_result[self.name]
        return metric_result[DEFAULT_SCORE_KEY]


class BLEU(TreevalMetric):
    """
    BLEU metric.

    Wrapper of the Hugging Face `"bleu" evaluation module
    <https://huggingface.co/spaces/evaluate-metric/bleu>`_.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(evaluate.load("bleu"), score_range=(0, 100), **kwargs)


class SacreBLEU(TreevalMetric):
    """
    SacreBLEU metric.

    Wrapper of the Hugging Face `"sacrebleu" evaluation module
    <https://huggingface.co/spaces/evaluate-metric/sacrebleu>`_.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(evaluate.load("sacrebleu"), score_range=(0, 100), **kwargs)


class ROUGE(TreevalMetric):
    """
    ROUGE metric.

    Wrapper of the Hugging Face `"rouge" evaluation module
    <https://huggingface.co/spaces/evaluate-metric/rouge>`_.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(evaluate.load("rouge"), **kwargs)


class MAUVE(TreevalMetric):
    """
    MAUVE metric.

    Wrapper of the Hugging Face `"mauve" evaluation module
    <https://huggingface.co/spaces/evaluate-metric/mauve>`_.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(evaluate.load("mauve"), **kwargs)


class METEOR(TreevalMetric):
    """
    METEOR metric.

    Wrapper of the Hugging Face `"meteor" evaluation module
    <https://huggingface.co/spaces/evaluate-metric/meteor>`_.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(evaluate.load("meteor"), **kwargs)


class Accuracy(TreevalMetric):
    """
    Accuracy metric for integers.

    Wrapper of the Hugging Face `"accuracy" evaluation module
    <https://huggingface.co/spaces/evaluate-metric/accuracy>`_.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(evaluate.load("accuracy"), **kwargs)


class Precision(TreevalMetric):
    """
    Precision metric.

    Wrapper of the Hugging Face `"precision" evaluation module
    <https://huggingface.co/spaces/evaluate-metric/precision>`_.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(evaluate.load("precision"), **kwargs)


class Recall(TreevalMetric):
    """
    Recall metric.

    Wrapper of the Hugging Face `"recall" evaluation module
    <https://huggingface.co/spaces/evaluate-metric/recall>`_.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(evaluate.load("recall"), **kwargs)


class F1(TreevalMetric):
    """
    F1 metric.

    Wrapper of the Hugging Face `"f1" evaluation module
    <https://huggingface.co/spaces/evaluate-metric/f1>`_.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(evaluate.load("f1"), **kwargs)


class MSE(TreevalMetric):
    """
    MSE metric.

    Wrapper of the Hugging Face `"mse" evaluation module
    <https://huggingface.co/spaces/evaluate-metric/mse>`_.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(evaluate.load("mse"), higher_is_better=False, **kwargs)


class ExactMatch(TreevalMetric):
    """
    Exact match metric, acting as a "type-agnostic" accuracy metric.

    This metric supports multiple types, but handles them slightly differently:

    * **strings** are passed to the Hugging Face `"exact_match" evaluation module
      <https://huggingface.co/spaces/evaluate-metric/exact_match>`_;
    * **floats** are compared using the numpy
      `is_close <https://numpy.org/doc/stable/reference/generated/numpy.isclose.html>`_
      method;
    * **integers** are compared using the numpy
      `equal <https://numpy.org/doc/stable/reference/generated/numpy.equal.html>`_
      method for faster execution;
    * **other types** are Pythonically compared using the ``__equal__`` magic method
      called with the ``==`` operator.

    :param rtol_float: relative tolerance argument to pass to the ``numpy.isclose``
        method (default: ``1e-5``).
    :param atol_float: relative tolerance argument to pass to the ``numpy.isclose``
        method (default: ``1e-8``).
    :param kwargs: keyword arguments to pass to the ``exact_match`` module for strings.
    """

    def __init__(
        self, rtol_float: float = 1e-5, atol_float: float = 1e-8, **kwargs
    ) -> None:
        self._string_acc = evaluate.load("exact_match")
        self._rtol_float = rtol_float
        self._atol_float = atol_float
        super().__init__(None, "exact_match", **kwargs)

    def _compute(
        self,
        predictions: Sequence[Any],
        references: Sequence[Any],
    ) -> dict:
        """
        Compute the metric score between pairs of references and predictions.

        :param predictions: list of predictions to evaluate.
        :param references: expected reference values.
        :return: the score as a dictionary. The absolute score, which is the average of
            the score of all individual pairs of reference/prediction, should be the
            value of an entry with the key being either "score" or the name of the
            metric.
        """
        if isinstance(predictions[0], str):
            return self._string_acc.compute(
                predictions=predictions, references=references, **self._kwargs
            )
        if isinstance(predictions, np.ndarray) or isinstance(
            predictions[0], (int, float)
        ):
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)
                references = np.array(references)
            if np.issubdtype(predictions.dtype, np.integer):
                res = np.equal(predictions, references)
            else:  # float
                res = np.isclose(
                    predictions, references, self._rtol_float, self._atol_float
                )
            return {"exact_match": np.count_nonzero(res) / len(predictions)}
        return {"exact_match": _exact_match_python(predictions, references)}


class BERTScore(TreevalMetric):
    """
    BERTScore metric.

    Wrapper of the Hugging Face `"bertscore" evaluation module
    <https://huggingface.co/spaces/evaluate-metric/bertscore>`_.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(evaluate.load("bertscore"), **kwargs)


class Perplexity(TreevalMetric):
    """
    Perplexity metric.

    Wrapper of the Hugging Face `"perplexity" evaluation module
    <https://huggingface.co/spaces/evaluate-metric/perplexity>`_.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(evaluate.load("perplexity"), **kwargs)


class RSquared(TreevalMetric):
    """
    R-squared metric.

    Wrapper of the Hugging Face `"r_squared" evaluation module
    <https://huggingface.co/spaces/evaluate-metric/r_squared>`_.
    """

    def __init__(self) -> None:  # module doesn't take kwargs
        super().__init__(evaluate.load("r_squared"))


class Levenshtein(TreevalMetric):
    """
    Levenshtein distance metrics.

    Wrapper of the Hugging Face `"Natooz/levenshtein"
    <https://huggingface.co/spaces/Natooz/levenshtein>`_ space.

    This metric uses the `Levenshtein <https://github.com/rapidfuzz/Levenshtein>`_
    Python package.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(evaluate.load("Natooz/levenshtein"), **kwargs)

    def _get_metric_score(self, metric_result: dict) -> float:
        """
        Return the absolute score value of the results returned by the metric.

        The Levenshtein ratio is used as score as already normalized.

        :param metric_result: metric results as a dictionary.
        :return: absolute metric score value as a floating point number.
        """
        return metric_result["levenshtein_ratio"]


def _exact_match_python(
    predictions: Sequence[Any],
    references: Sequence[Any],
) -> float:
    return len([0 for pred, ref in zip(predictions, references) if pred == ref]) / len(
        predictions
    )
