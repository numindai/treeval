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
from evaluate import EvaluationModule
from Levenshtein import distance

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
    :param score_range:
    :param higher_is_better:
    """

    def __init__(
        self,
        module: Callable | EvaluationModule | None,
        name: str | None = None,
        score_range: tuple[float | int, float | int] = (0, 1),
        higher_is_better: bool = True,
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

    def compute(
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
        if self._is_hf_module:
            return self._module.compute(predictions=predictions, references=references)
        return self._module(predictions=predictions, references=references)

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
        if self.name in metric_result:
            return metric_result[self.name]
        return metric_result[DEFAULT_SCORE_KEY]


class BLEU(TreevalMetric):
    """BLEU, wrapper of the Hugging Face evaluation module."""

    def __init__(self) -> None:
        super().__init__(evaluate.load("bleu"), score_range=(0, 100))


class SacreBLEU(TreevalMetric):
    """SacreBLEU, wrapper of the Hugging Face evaluation module."""

    def __init__(self) -> None:
        super().__init__(evaluate.load("sacrebleu"), score_range=(0, 100))


class ROUGE(TreevalMetric):
    """ROUGE, wrapper of the Hugging Face evaluation module."""

    def __init__(self) -> None:
        super().__init__(evaluate.load("rouge"))


class MAUVE(TreevalMetric):
    """MAUVE, wrapper of the Hugging Face evaluation module."""

    def __init__(self) -> None:
        super().__init__(evaluate.load("mauve"))


class METEOR(TreevalMetric):
    """METEOR, wrapper of the Hugging Face evaluation module."""

    def __init__(self) -> None:
        super().__init__(evaluate.load("meteor"))


class Accuracy(TreevalMetric):
    """Precision, wrapper of the Hugging Face evaluation module."""

    def __init__(self) -> None:
        super().__init__(evaluate.load("accuracy"))


class Precision(TreevalMetric):
    """Precision, wrapper of the Hugging Face evaluation module."""

    def __init__(self) -> None:
        super().__init__(evaluate.load("precision"))


class Recall(TreevalMetric):
    """Recall, wrapper of the Hugging Face evaluation module."""

    def __init__(self) -> None:
        super().__init__(evaluate.load("recall"))


class F1(TreevalMetric):
    """F1, wrapper of the Hugging Face evaluation module."""

    def __init__(self) -> None:
        super().__init__(evaluate.load("f1"))


class MSE(TreevalMetric):
    """MSE, wrapper of the Hugging Face evaluation module."""

    def __init__(self) -> None:
        super().__init__(evaluate.load("mse"), higher_is_better=False)


class ExactMatch(TreevalMetric):
    """Exact match, wrapper of the Hugging Face evaluation module."""

    def __init__(self) -> None:
        super().__init__(evaluate.load("exact_match"))


class BERTScore(TreevalMetric):
    """BERTScore, wrapper of the Hugging Face evaluation module."""

    def __init__(self) -> None:
        super().__init__(evaluate.load("bertscore"))


class Perplexity(TreevalMetric):
    """Perplexity, wrapper of the Hugging Face evaluation module."""

    def __init__(self) -> None:
        super().__init__(evaluate.load("perplexity"))


class RSquared(TreevalMetric):
    """RSquared, wrapper of the Hugging Face evaluation module."""

    def __init__(self) -> None:
        super().__init__(evaluate.load("r_squared"))


class Levenshtein(TreevalMetric):
    """
    Levenshtein distance, based on the `"Levenshtein" PyPi package <https://github.com/rapidfuzz/Levenshtein>`_.

    This metric, also called "edit distance", measures the number of editions, deletions
    and additions to perform on a string so that it becomes identical to a second one.
    This metric returns two entries:

    * ``levenshtein``: corresponds to an indel string similarity ratio normalized
      between [0, 1]. It is equivalent to the ``Levenshtein.ratio`` method.
    * ``edit_distance``: the actual average edit distance between the references and
      predictions.
    """

    def __init__(self) -> None:
        super().__init__(None, "levenshtein")

    def _levenshtein_distance(
        self,
        predictions: Sequence[Any],
        references: Sequence[Any],
    ) -> dict:
        results, ratios = [], []
        for prediction, reference in zip(predictions, references):
            edit_distance = distance(prediction, reference)
            results.append(edit_distance)
            ratios.append(edit_distance / (len(prediction) + len(reference)))

        return {
            self.name: 1 - sum(ratios) / len(ratios),
            "edit_distance": sum(results) / len(results),
        }


class BooleanAccuracy(TreevalMetric):
    """Check the equality of booleans, yielding 1 if they are equal, 0 otherwise."""

    def __init__(self) -> None:
        super().__init__(_boolean_accuracy, "boolean_accuracy")


def _boolean_accuracy(
    predictions: Sequence[bool],
    references: Sequence[bool],
) -> dict[str, float]:
    return {
        "boolean_accuracy": len(
            [0 for pred, ref in zip(predictions, references) if pred == ref]
        )
        / len(predictions)
    }
