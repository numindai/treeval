"""Utils methods."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Any

    import numpy as np


def flatten_nested_dict(
    dictionary: dict, parent_key: str = "", separator: str = "."
) -> list[tuple[str, Any]]:
    """
    Flatten a dictionary (recursively) into a list of tuples of key and value.

    All the keys are expected to be strings.

    :param dictionary: dictionary to flatten.
    :param parent_key: parent key string when the method is called recursively.
    :param separator: a string separating the keys of nested dictionaries.
    :return: a list of tuples with key value pairs.
    """
    items = []

    if isinstance(dictionary, list):
        # handle empty list case
        if len(dictionary) == 0:
            items.append((parent_key, ""))

        for index, value in enumerate(dictionary):
            new_key = (
                f"{parent_key}<{index}>" if parent_key else str(index)
            )  # with list indexing
            # new_key = f"{parent_key}" if parent_key else str(index) # no list indexing
            items.extend(flatten_nested_dict(value, new_key, separator))

    elif isinstance(dictionary, dict):
        for key, value in dictionary.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key
            items.extend(flatten_nested_dict(value, new_key, separator))

    else:
        # items.append(f"{parent_key}={obj}")
        items.append((parent_key, dictionary))

    return items


def merge_dicts(dict1: dict, dict2: dict, discard_none: bool = False) -> dict:
    """
    Recursively merge two dictionaries of the same format.

    Primitive types are merged into lists.

    :param dict1: The first dictionary.
    :param dict2: The second dictionary.
    :param discard_none: if provided ``True``, ``None`` values will be discarded from
        the final merged dictionary.
    :return: The merged dictionary.
    """
    merged = {}

    # Get all unique keys from both dictionaries
    keys = set(dict1.keys()).union(set(dict2.keys()))

    for key in keys:
        if key in dict1 and key in dict2:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                # Both values are dictionaries, merge them recursively
                merged[key] = merge_dicts(dict1[key], dict2[key])
            elif isinstance(dict1[key], list) and isinstance(dict2[key], list):
                # Both values are lists, concatenate them
                merged[key] = dict1[key] + dict2[key]
            else:
                # Merge primitive types into lists
                merged[key] = []
                if dict1[key] is not None or not discard_none:
                    if isinstance(dict1[key], list):
                        merged[key].extend(dict1[key])
                    else:
                        merged[key].append(dict1[key])
                if dict2[key] is not None or not discard_none:
                    if isinstance(dict2[key], list):
                        merged[key].extend(dict2[key])
                    else:
                        merged[key].append(dict2[key])
        elif key in dict1:
            # Key only in dict1
            merged[key] = dict1[key]
        else:
            # Key only in dict2
            merged[key] = dict2[key]

    return merged


def count_dictionary_nodes(dictionary: dict, only_leaves: bool = False) -> int:
    """
    Count the total number of nodes within a dictionary tree, recursively.

    :param dictionary: dictionary to inspect.
    :param only_leaves: will only count the leaves of the dictionary, excluding parent
        nodes of nested dictionaries.
    :return: total number of nodes within ``dictionary`` including nested dictionaries.
    """
    num_nodes = 0
    for value in dictionary.values():
        if isinstance(value, dict):
            num_nodes += count_dictionary_nodes(value, only_leaves)
            if not only_leaves:
                num_nodes += 1
        else:
            num_nodes += 1
    return num_nodes


def get_unique_leaf_values(dictionary: dict) -> set[str]:
    """
    Return the set of unique leaf values within a dictionary.

    All leaf values must be hashable.

    :param dictionary: dictionary to inspect.
    :return: set of unique leaf values within the dictionary.
    """
    metrics = set()
    for node_value in dictionary.values():
        if isinstance(node_value, dict):
            metrics |= get_unique_leaf_values(node_value)
        elif isinstance(node_value, set):
            metrics |= node_value
        else:
            metrics.add(node_value)
    return metrics


def compute_matching_from_score_matrix(
    score_matrix: np.ndarray, maximize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Determine the pairs of references/predictions items from two lists.

    It corresponds to a form of `assignment problem <https://en.wikipedia.org/wiki/Assignment_problem>`_,
    which can be solved by computing the minimum/maximum matching of a weighted
    bipartite graph whose edges weights are the metrics scores. This method simply uses
    scipy's `linear_sum_assignment <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html>`_
    method.

    :param score_matrix: metric scores for all combinations of pairs of ref/pred.
    :param maximize: calculate the matching in order to maximize the sum of the values
        of the edges, i.e. maximize the metric scores. Provide ``False`` for metrics
        with lower scores being better. (default: ``True``)
    :return: The alignment score and the aligned indices.
    """
    return linear_sum_assignment(score_matrix, maximize=maximize)


def load_json_files(
    json_files_paths: Sequence[Path], skip_json_parsing_errors: bool = False
) -> list[dict]:
    """
    Load JSON files and decode them into dictionaries.

    :param json_files_paths: paths of the JSON files to load.
    :param skip_json_parsing_errors: whether to skip ``JSONDecodeError`` parsing errors.
        If provided ``True``, this method will raise an exception if any parsing error
        is encountered. (default: ``False``)
    :return: contents of the JSON files as a list of dictionaries.
    """
    data = []

    for file_path in json_files_paths:
        try:
            with file_path.open() as file:
                data_ = json.load(file)
            data.append(data_)
        except json.JSONDecodeError as e:
            if not skip_json_parsing_errors:
                raise e

    return data
