"""Treeval tree evaluation method."""

from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

import numpy as np

from .utils import (
    compute_matching_from_score_matrix,
    count_dictionary_nodes,
    merge_dicts,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from .metrics import TreevalMetric


# Precision/Recall/F1 are appended "nodes"/"leaves" to avoid confusion with the same
# metrics being computed on the leaves results. When aggregating results, this could
# mess up results.
PRECISION_NODE_KEY = "precision_node"
RECALL_NODE_KEY = "recall_node"
F1_NODE_KEY = "f1_node"
PRECISION_LEAF_KEY = "precision_leaf"
RECALL_LEAF_KEY = "recall_leaf"
F1_LEAF_KEY = "f1_leaf"
_PRF_METRIC_NAMES = {
    PRECISION_NODE_KEY,
    RECALL_NODE_KEY,
    F1_NODE_KEY,
    PRECISION_LEAF_KEY,
    RECALL_LEAF_KEY,
    F1_LEAF_KEY,
}


def treeval(
    predictions: Sequence[dict],
    references: Sequence[dict],
    schema: dict,
    metrics: dict[str, TreevalMetric],
    tree_metrics: dict,
    aggregate_results_per_metric: bool = False,
    aggregate_results_per_leaf_type: bool = False,
    hierarchical_averaging: bool = False,
) -> dict:
    """
    Treeval evaluation method.

    :param predictions: list of dictionary predictions.
    :param references: list of dictionary references.
    :param schema: schema of the tree as a dictionary specifying each leaf type. The
        references must all follow this exact tree structure, while the predictions can
        have mismatching branches which will impact the tree precision/recall/f1 scores
        returned by the method.
    :param metrics: metrics to use to evaluate the leaves of the trees.
    :param tree_metrics: dictionary with the same schema/structure as ``schema``
        specifying at each leaf the set of metrics to use for evaluate them, referenced
        by their names as provided in ``metrics``. See the
        :py:func:`treeval.create_tree_metrics` method to easily create this argument by
        mapping leaf types to metrics.
    :param aggregate_results_per_metric: averages the final results per metric. Enabling
        this option will call the :py:func:`treeval.aggregate_results_per_metric` method
        on the final tree results. (default: ``False``)
    :param aggregate_results_per_leaf_type: averages the final results per node type.
        Enabling this option will call the
        :py:func:`treeval.aggregate_results_per_leaf_type` method on the final tree
        results. (default: ``False``)
    :param hierarchical_averaging: argument to be used when one of the
        ``aggregate_results_per_metric`` or ``aggregate_results_per_leaf_type`` options
        is enabled. See the documentation of the
        :py:func:`treeval.aggregate_results_per_metric` and
        :py:func:`treeval.aggregate_results_per_leaf_type` for the complete explanation.
        (default: ``False``)
    :return: the metrics results. The returned dictionary will have the same tree
        structure as the provided ``schema`` if ``aggregate_results_per_metric`` or
        ``aggregate_results_per_leaf_type`` are disabled, otherwise it will map metrics
        and/or leaf types to average results over all leaves.
    """
    # Check number of predictions/references
    if len(predictions) != len(references):
        msg = "The number of predictions must be equal to the number of references."
        raise ValueError(msg)

    # Recursively parses the schema and computes the metrics scores at the leaves
    results = _recursive_parse(predictions, references, schema, metrics, tree_metrics)

    # Aggregate per metric and/or type
    if aggregate_results_per_metric:
        if aggregate_results_per_leaf_type:
            warn(
                "treeval: you set both the `aggregate_results_per_metric` and `"
                "aggregate_results_per_leaf_type` arguments as `True`. The results can "
                "be summarized with one method only. `aggregate_results_per_metric` "
                "will take precedence.",
                stacklevel=2,
            )
        return _aggregate_results_per_metric(
            results,
            tree_metrics,
            metrics,
            hierarchical_averaging=hierarchical_averaging,
        )
    if aggregate_results_per_leaf_type:
        return _aggregate_results_per_leaf_type(
            results, schema, metrics, hierarchical_averaging=hierarchical_averaging
        )
    return results


def _recursive_parse(
    predictions: Sequence[dict | Any],
    references: Sequence[dict | Any],
    schema: dict | str,
    metrics: dict[str, TreevalMetric],
    tree_metrics: dict | list[str],
    pr_cache: list[int] | None = None,
) -> dict | list[list[dict]]:
    # Returns the evaluation metric score as a tree with metrics results averaged
    # over the same leaf values of all predictions/references pairs.
    # This method is recursive and supports nested dictionaries and lists of items of
    # specific types, including lists of nested dictionaries.

    # Leaf (basic case) --> compute metrics on values
    if not isinstance(predictions[0], (dict, list)):  # -> {metric_name: score}
        return {
            metric_name: metrics[metric_name].compute(predictions, references)
            for metric_name in tree_metrics
        }

    # List --> match the elements in the lists of each reference/prediction pair
    # Lists of choice do not fall in this condition as they are evaluated as single
    # leaves in the above if condition.
    # TODO handle multilabel classification
    if isinstance(predictions[0], list):
        # mean of aligned element match scores

        # Computes metrics on all combinations of ref/pred items within the lists,
        # independently. Batching is not possible here as the metrics are expected to
        # return the mean over all the items within the batch, whereas we need all the
        # individual scores in order to perform alignment to keep the best scores.
        # If getting individual scores is possible, batching would be feasible by
        # processing the (n * m) combinations in parallel for the items in the lists of
        # each pair of ref/pred. Batching multiple pairs of ref/pred is not possible as
        # lists usually have different sequence lengths, or it would require to keep
        # track of each number of combinations (n * m) per ref/pred pair.
        # If the items of the list are dictionaries, `tree_metrics` is a dictionary
        # with sets of metrics names as leaf values. Otherwise, it is a set of
        # metrics names.
        is_list_of_dicts, _idx = False, 0
        while _idx < len(predictions):
            if len(predictions[0]) > 0:
                is_list_of_dicts = isinstance(predictions[_idx][0], dict)
                break
        __metrics_set = (
            __get_unique_metrics_from_tree_metrics(tree_metrics)
            if is_list_of_dicts
            else tree_metrics
        )
        results = {metric_name: [] for metric_name in __metrics_set}
        for pred, ref in zip(predictions, references):
            # Create the matrices storing the metrics results
            # `metrics_results` stores the "raw"/complete results of each metric.
            # `metrics_scores` stores the score results of each metric.
            # If the items of the list are dictionaries, `tree_metrics` is a dictionary
            # with sets of metrics names as leaf values. Otherwise, it is a set of
            # metrics names.
            metrics_results = [[] for _ in range(len(ref))]  # (n,m,{name: res})
            metrics_scores = {  # {metric_name: (n,m)}, score for assignment only
                metric: [[] for _ in range(len(ref))] for metric in __metrics_set
            }
            # TODO count difference of lengths between two lists -> find way to penalize
            #  - penalty in the metrics? inject `n` worst score (n = abs(diff)))
            #  - report additional measurement on the expected/prediction num of items

            # Compute metrics, unbatched as we need to match the references/predictions.
            # For simplicity reasons, the precision/recall/f1 of dictionary items (when
            # list of dictionaries) are not computed. The precision/recall/f1 returned
            # by treeval stops at the leaves, lists of dictionaries are considered as
            # leaves whatever their depths may be.
            # (n,m) --> {metric_name: metric_results_dict}
            for idx, ref_i in enumerate(ref):
                for pred_i in pred:
                    score = _recursive_parse(
                        [pred_i],
                        [ref_i],
                        schema[0],  # provides either a type or a dict (list of dicts)
                        metrics,
                        tree_metrics,  # already list of metric names
                        [0] * 6,  # mocking the pr_cache so that it doesn't return
                    )  # the precision/recall/scores in the score dictionary (dicts).
                    # If the items are dictionaries, we aggregate the tree results per
                    # metric to be able to assign pairs of references/predictions.
                    if is_list_of_dicts:
                        score = _aggregate_results_per_metric(
                            score, tree_metrics, metrics
                        )
                    metrics_results[idx].append(score)

                    # Adds the score to the matrix.
                    # `_aggregate_results_per_metric` (list of dicts) already calls
                    # `get_metric_score`, it should only be called in other cases.
                    for metric_name, metric_score in score.items():
                        metrics_scores[metric_name][idx].append(
                            metrics[metric_name].get_metric_score(metric_score)
                            if not is_list_of_dicts
                            else metric_score
                        )

            # Normalize metrics scores matrices between within [0, 1]
            # TODO support non-unidirectional metrics (when not higher/lower better)
            for metric_name in metrics_scores:
                metric_score_array = np.array(metrics_scores[metric_name])
                if len(metrics_scores) > 1:  # no need if only one metric
                    if metrics[metric_name].score_range != (0, 1):
                        low_bound, high_bound = metrics[metric_name].score_range
                        metric_score_array = (
                            metric_score_array.clip(low_bound, high_bound) - low_bound
                        ) / (high_bound - low_bound)
                    if not metrics[metric_name].higher_is_better:
                        metric_score_array = (
                            np.ones_like(metric_score_array) - metric_score_array
                        )
                metrics_scores[metric_name] = metric_score_array

            # Average the normalized arrays
            if len(metrics_scores) > 1:  # [(n,m)] --> (s,n,m) --> (n,m)
                metrics_scores_average = np.mean(
                    np.stack(list(metrics_scores.values()), axis=0), axis=0
                )
            else:
                metrics_scores_average = next(iter(metrics_scores.values()))

            # Computes the assignment/pairs of reference/prediction items
            pairs_idx = compute_matching_from_score_matrix(metrics_scores_average)
            for ref_idx, pred_idx in zip(*pairs_idx):
                for metric_name in metrics_scores:
                    results[metric_name].append(
                        metrics_scores[metric_name][ref_idx][pred_idx]
                    )

        # Average, return {metric_res} only
        # We can't return other metric elements as we didn't batch the computations, and
        # we can't assume how to average them, so we only cover the score.
        return {
            metric_name: {metric_name: sum(metric_scores) / len(metric_scores)}
            for metric_name, metric_scores in results.items()
        }

    # Dictionary --> recursive parsing
    results = {}
    root_node = False
    if pr_cache is None:  # root node
        root_node = True
        pr_cache = [0] * 6  # counts TT/TP/FN (nodes) + TP/FN/FP (leaf)
    # Counts the total number of nodes (at current branch/depth) before parsing the
    # tree batching all preds/refs
    pr_cache[0] += sum(len(pred) for pred in predictions)
    for node_name, node_type in schema.items():
        # Gathers pairs of refs/preds that are both present.
        # Count precision/recall scores on nodes/leaves
        node_predictions, node_references = [], []
        for pred, ref in zip(predictions, references):
            # Node is in both
            if node_name in pred:
                pred_leaf_val = pred[node_name]
                ref_leaf_val = ref[node_name]

                # TP: leaf in both ref and pred
                if pred_leaf_val is not None and ref_leaf_val is not None:
                    node_predictions.append(pred_leaf_val)
                    node_references.append(ref_leaf_val)  # must always be in ref
                    if not isinstance(node_type, dict):
                        pr_cache[3] += 1  # increments leaf TP count

                # FN: leaf in ref missing from pred
                # leaf prf are only computed on leaves, excluding intermediates nodes
                elif pred_leaf_val is None and ref_leaf_val is not None:
                    pr_cache[4] += (
                        count_dictionary_nodes(node_type, only_leaves=True)
                        if isinstance(node_type, dict)
                        else 1
                    )
                    if isinstance(node_type, dict):
                        pr_cache[2] += count_dictionary_nodes(node_type)  # node FN

                # FP: leaf in pred not present in ref (unexpected)
                elif pred_leaf_val is not None and ref_leaf_val is None:
                    if isinstance(pred_leaf_val, dict):
                        pr_cache[5] += count_dictionary_nodes(
                            pred_leaf_val, only_leaves=True
                        )
                        pr_cache[0] += count_dictionary_nodes(
                            pred_leaf_val
                        )  # total nodes
                    else:
                        pr_cache[5] += 1

                pr_cache[1] += 1  # increments nodes TP count

            # node false negative += total number of children nodes + 1 (current node)
            else:
                pr_cache[2] += (
                    count_dictionary_nodes(node_type) + 1
                    if isinstance(node_type, dict)
                    else 1
                )
                if isinstance(node_type, dict):
                    pr_cache[4] += count_dictionary_nodes(node_type, only_leaves=True)

        if len(node_predictions) > 0:
            results[node_name] = _recursive_parse(
                node_predictions,
                node_references,
                node_type,
                metrics,
                tree_metrics[node_name],
                pr_cache,
            )
        else:
            results[node_name] = None
            # TODO otherwise None? --> handle None cases + docs

    # Compute the precision, recall and f1 scores of the predicted nodes/leaves
    if root_node:
        # tp + fn should be equal to len(references) * count_dictionary_nodes(schema)
        # there is no tn for nodes, for leaves yes (`None` leaf values in references)
        total_num_nodes, tp_node, fn_node, tp_leaf, fn_leaf, fp_leaf = pr_cache

        # Add node precision/recall/f1 scores to the results
        fp_node = total_num_nodes - tp_node  # predicted nodes not in references
        precision_node = tp_node / (tp_node + fp_node)
        recall_node = tp_node / (tp_node + fn_node)
        results[PRECISION_NODE_KEY] = precision_node
        results[RECALL_NODE_KEY] = recall_node
        results[F1_NODE_KEY] = __compute_f1(precision_node, recall_node)

        # Add leaf precision/recall/f1 scores to the results
        # cases where no None at all --> 1 scores
        # num_leaves = tn_leaf + tp_leaf + fn_leaf + fp_leaf
        precision_leaf = tp_leaf / (tp_leaf + fp_leaf) if tp_leaf + fp_leaf > 0 else 1
        recall_leaf = tp_leaf / (tp_leaf + fn_leaf) if tp_leaf + fn_leaf > 0 else 1
        results[PRECISION_LEAF_KEY] = precision_leaf
        results[RECALL_LEAF_KEY] = recall_leaf
        results[F1_LEAF_KEY] = __compute_f1(precision_leaf, recall_leaf)

    return results


def __compute_f1(precision: float, recall: float) -> float:
    return (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )


def create_tree_metrics(
    schema: dict,
    leaves_metrics: dict | None = None,
    types_metrics: dict[str | tuple, Sequence[str]] | None = None,
    exclusive_leaves_types_metrics: bool = False,
) -> dict:
    """
    Create the ``tree_metrics`` of a schema from specific leaf names and/or types.

    An error is raised if a leaf cannot be evaluated by any of the metrics provided in
    ``leaves_metrics`` and ``types_metrics``.

    :param schema: structure of the tree as a dictionary specifying each leaf type.
    :param leaves_metrics: dictionary with the same tree structure as the provided
        ``schema`` specifying the metrics to compute for specific leaves.
        (default: ``None``)
    :param types_metrics: dictionary mapping the types specified in the provided
        ``schema`` to the metrics to compute for the leaves of these types. All types
        names must be strings, except the empty tuple ``()`` which is used for choice
        lists. (default: ``None``)
    :param exclusive_leaves_types_metrics: an option allowing to make the metrics
        specified in ``leaves_metrics`` to be exclusive to certain leaves, excluding the
        metrics that should cover them specified in the ``types_metrics`` argument.
        Example: for the ``schema`` ``{"foo": "integer"}``, ``leaves_metrics``
        ``{"foo": "accuracy"}`` and ``types_metrics`` ``{"foo": "mse"}``, if this option
        is enabled, the method will return the ``{"foo": {"accuracy"}}`` tree metrics as
        the metric specified in ``leaves_metrics`` will take precedence. Otherwise, the
        method will return ``{"foo": {"accuracy", "mse"}}``, combining the metrics from
        the two arguments. This option can be especially useful when some specific
        leaves are expected to be evaluated with specific metrics. (default: ``False``)
    :return: tree identical to ``schema`` where leaf values reference to the **set** of
        names of metrics to use to evaluate them.
    """
    # Safety check
    if not leaves_metrics:
        leaves_metrics = {}
    if not types_metrics:
        types_metrics = {}

    tree_metrics = {}
    for node_name, node_type in schema.items():
        if isinstance(node_type, list):
            node_type_tmp = node_type[0] if len(node_type) == 1 else ()  # else choice
        else:
            node_type_tmp = node_type
        if isinstance(node_type_tmp, dict):
            tree_metrics[node_name] = create_tree_metrics(
                node_type_tmp,
                leaves_metrics.get(node_name, {}),
                types_metrics,
                exclusive_leaves_types_metrics,
            )
        else:
            leaf_metrics = leaves_metrics.get(node_name, []).copy()
            if not exclusive_leaves_types_metrics or len(leaf_metrics) == 0:
                leaf_metrics += types_metrics.get(node_type_tmp, [])
            for metric_name in leaf_metrics:
                if metric_name in _PRF_METRIC_NAMES:
                    msg = (
                        f"The `{metric_name}` metric name cannot be used, please rename"
                        f" it. Treeval forbids the use of {_PRF_METRIC_NAMES} "
                        "metric names as they are used to computed on the nodes at the "
                        "tree-level."
                    )
                    raise ValueError(msg)
            if len(leaf_metrics) == 0:
                msg = (
                    "Incompatible schema/leaves_metrics/types_metrics provided. The "
                    f"leaf `{node_name}` is not covered by any metric and cannot be "
                    "evaluated."
                )
                raise ValueError(msg)
            tree_metrics[node_name] = set(leaf_metrics)

    return tree_metrics


def __get_unique_metrics_from_tree_metrics(tree_metrics: dict) -> set[str]:
    """
    Get the set of unique metric names present in a "tree metrics".

    :param tree_metrics: dictionary with the same schema/structure as ``results``
        specifying at each leaf the set of metrics to use for evaluate them, referenced
        by their names as found in the ``results``.
    :return: set of the names of the unique metrics present in ``tree_metrics``.
    """
    metrics_names = set()
    for node_value in tree_metrics.values():
        if isinstance(node_value, dict):
            metrics_names |= __get_unique_metrics_from_tree_metrics(node_value)
        else:
            metrics_names |= node_value
    return metrics_names


def _aggregate_results_per_metric(
    results: dict,
    tree_metrics: dict,
    metrics: dict[str, TreevalMetric],
    hierarchical_averaging: bool = False,
) -> dict[str, float]:
    """
    Aggregate the tree treeval results per metric.

    This method will return a single-depth dictionary mapping each metric of the
    provided ``results`` to the average of its scores found within the results tree.

    :param results: non-aggregated results from the :py:func:`treeval.treeval` method.
    :param tree_metrics: dictionary with the same schema/structure as ``results``
        specifying at each leaf the set of metrics to use for evaluate them, referenced
        by their names as found in the ``results``.
    :param hierarchical_averaging: averages the metrics scores at each branch depth. If
        this option is enabled, the scores of the metrics will be averaged at each
        branch of nested dictionaries. These averages will be included in the metrics
        scores of the parent node as a single value, as opposed to including all the
        score metrics of the branch to compute the average of the parent node.
        This option allows to give more importance in the final results to the scores of
        the leaves at lower depths, closer to the root.
        Example: ``{"n1": 0, "n2": 1, "n3": {"n4": 0, "n5": 1}}`` represents the scores
        of a given metric for this dictionary structure. If hierarchical averaging is
        enabled, the scores for the metric (root node) will be computed from scores of
        the ``"n1"``, ``"n2"`` nodes and the average of the nodes within the ``"n3"``
        branch, i.e. average of ``[0, 0, 0.5]``. If hierarchical averaging is enabled,
        the average is computed from all the scores in the tree with no distinction
        towards the depths of the leaves, e.g. ``[0, 0, 0, 1]`` in the previous
        examples. (default: ``False``)
    :return: single-depth dictionary mapping each metric of the provided ``results`` to
        the average of its scores found within the results tree.
    """
    # Gather the scores of all individual metrics in all leaves.
    metrics_results = __aggregate_results_per_metric(
        results, tree_metrics, metrics, hierarchical_averaging=hierarchical_averaging
    )

    # If hierarchical averaging is enabled, the averaging is already done in the child
    # method. Otherwise, it returned the list of all scores that we need to average.
    if not hierarchical_averaging:
        metrics_results = {
            metric_name: sum(scores) / len(scores)
            for metric_name, scores in metrics_results.items()
        }

    # Re-include the tree precision/recall/f1 scores in the results to return.
    # These elements might not be in the `results` when this method is called to
    # aggregate results in lists of dictionaries before assignment.
    for prf_key in _PRF_METRIC_NAMES:
        if results_prf := results.get(prf_key):
            metrics_results[prf_key] = results_prf

    return metrics_results


def __aggregate_results_per_metric(
    results: dict,
    tree_metrics: dict,
    metrics: dict[str, TreevalMetric],
    hierarchical_averaging: bool = False,
) -> dict[str, float | list[float]]:
    # Same as _aggregate_results_per_metric but recursive and discarding
    # precision/recall/f1 entries that are added at the end.

    # Gather scores per metric name
    metrics_results = {}  # {metric_name: [results]}
    for node_name, value in tree_metrics.items():
        results_node = results[node_name]
        if isinstance(value, dict):
            # Need to check that the keys of the branch results are all already present
            # in the metrics_results dictionary before merging them
            branch_results = __aggregate_results_per_metric(
                results_node,
                value,
                metrics,
                hierarchical_averaging=hierarchical_averaging,
            )
            for key_branch in branch_results:
                if key_branch not in metrics_results:
                    metrics_results[key_branch] = []
            metrics_results = merge_dicts(metrics_results, branch_results)
        else:
            for metric_name, metric_results in results_node.items():
                if metric_name not in metrics_results:
                    metrics_results[metric_name] = []
                metrics_results[metric_name].append(
                    metrics[metric_name].get_metric_score(metric_results)
                )

    # Averages the scores of the current branch if hierarchical averaging
    if hierarchical_averaging:
        return {
            metric_name: sum(scores) / len(scores)
            for metric_name, scores in metrics_results.items()
        }
    return metrics_results


def _aggregate_results_per_leaf_type(
    results: dict,
    schema: dict,
    metrics: dict[str, TreevalMetric],
    hierarchical_averaging: bool = False,
) -> dict[dict[str, float]]:
    """
    Aggregate the tree treeval results per leaf type.

    This method will return a single-depth dictionary mapping each metric of the
    provided ``results`` to the average of its scores found within the results tree.

    :param results: non-aggregated results from the :py:func:`treeval.treeval` method.
    :param schema: structure of the tree as a dictionary specifying each leaf type.
    :param hierarchical_averaging: averages the metrics scores at each branch depth. If
        this option is enabled, the scores of the metrics will be averaged at each
        branch of nested dictionaries. These averages will be included in the metrics
        scores of the parent node as a single value, as opposed to including all the
        score metrics of the branch to compute the average of the parent node.
        This option allows to give more importance in the final results to the scores of
        the leaves at lower depths, closer to the root.
        Example: ``{"n1": 0, "n2": 1, "n3": {"n4": 0, "n5": 1}}`` represents the scores
        of a given metric for this dictionary structure. If hierarchical averaging is
        enabled, the scores for the metric (root node) will be computed from scores of
        the ``"n1"``, ``"n2"`` nodes and the average of the nodes within the ``"n3"``
        branch, i.e. average of ``[0, 0, 0.5]``. If hierarchical averaging is enabled,
        the average is computed from all the scores in the tree with no distinction
        towards the depths of the leaves, e.g. ``[0, 0, 0, 1]`` in the previous
        examples. (default: ``False``)
    :return: single-depth dictionary mapping each leaf type of the provided ``results``
        to the average of its scores found within the results tree.
    """
    # Gather the scores of all individual metrics in all leaves.
    results_types = __aggregate_results_per_leaf_type(
        results, schema, metrics, hierarchical_averaging=hierarchical_averaging
    )

    # If hierarchical averaging is enabled, the averaging is already done in the child
    # method. Otherwise, it returned the list of all scores that we need to average.
    if not hierarchical_averaging:
        results_types = {
            type_name: {
                met: sum(scores) / len(scores) for met, scores in metrics_scores.items()
            }
            for type_name, metrics_scores in results_types.items()
        }

    # Re-include the tree precision/recall/f1 scores in the results to return
    for prf_key in _PRF_METRIC_NAMES:
        if results_prf := results.get(prf_key):
            results_types[prf_key] = results_prf

    return results_types


def __aggregate_results_per_leaf_type(
    results: dict,
    schema: dict,
    metrics: dict[str, TreevalMetric],
    hierarchical_averaging: bool = False,
) -> dict[dict[str, float | list[float]]]:
    # Same as _aggregate_results_per_leaf_type but recursive and discarding
    # precision/recall/f1 entries that are added at the end.

    # Gather scores per metric name
    results_types = {}  # {type: {metric: [results]}}
    for node_name, type_name in schema.items():
        results_node = results[node_name]
        if isinstance(type_name, dict):
            # Need to check that the keys of the branch results are all already present
            # in the metrics_results dictionary before merging them
            branch_results = __aggregate_results_per_leaf_type(
                results_node,
                type_name,
                metrics,
                hierarchical_averaging=hierarchical_averaging,
            )
            for key_branch, metrics_branch in branch_results.items():
                if key_branch not in results_types:
                    results_types[key_branch] = {}
                for key_metric_branch in metrics_branch:
                    if key_metric_branch not in results_types[key_branch]:
                        results_types[key_branch][key_metric_branch] = []
            results_types = merge_dicts(results_types, branch_results)
        else:
            if type_name not in results_types:
                results_types[type_name] = {}
            for metric_name, metric_results in results_node.items():
                if metric_name not in results_types[type_name]:
                    results_types[type_name][metric_name] = []
                results_types[type_name][metric_name].append(
                    metrics[metric_name].get_metric_score(metric_results)
                )

    # Compute the mean of each type scores
    if hierarchical_averaging:
        return {
            type_name: {
                met: sum(scores) / len(scores) for met, scores in metrics_scores.items()
            }
            for type_name, metrics_scores in results_types.items()
        }
    return results_types
