"""Tests for utils methods."""

import pytest
from treeval.utils import count_dictionary_nodes, merge_dicts

DICTIONARIES_NUM_NODES = [
    ({}, 0),
    ({"node1": 1, "node2": 2}, 2),
    ({"node1": 1, "node2": {"nested1": 1}}, 3),
    ({"node1": 1, "node2": {"nested11": 1, "nested12": 2}}, 4),
    ({"node1": 1, "node2": {"nested11": 1, "nested12": {"nest": 0, "a": 0}}}, 6),
]


def test_merge_results() -> None:
    res1 = {
        "exact_match": {"precision": 1.0, "recall": 0.5, "f1": 0.75},
        "bertscore": {"precision": 0.5, "recall": 0.5, "f1": 0.5},
    }
    res2 = {
        "exact_match": {"precision": 0.5, "recall": 0.5, "f1": 0.5},
        "bertscore": {"precision": 0.5, "recall": 0.5, "f1": 0.5},
    }

    res = merge_dicts(res1, res2)
    assert res == {
        "exact_match": {
            "precision": [1.0, 0.5],
            "recall": [0.5, 0.5],
            "f1": [0.75, 0.5],
        },
        "bertscore": {"precision": [0.5, 0.5], "recall": [0.5, 0.5], "f1": [0.5, 0.5]},
    }


@pytest.mark.parametrize(("dictionary", "expected_num_nodes"), DICTIONARIES_NUM_NODES)
def test_count_dictionary_nodes(dictionary: dict, expected_num_nodes: int) -> None:
    assert count_dictionary_nodes(dictionary) == expected_num_nodes
