"""Tests for the alignment method."""

from __future__ import annotations

import numpy as np
import pytest
from treeval.utils import compute_matching_from_score_matrix

TEST_CASES = [
    # Basic cases
    ([[1]], ([0], [0]), True),
    ([[1, 1]], ([0], [0]), True),
    ([[1], [1]], ([0], [0]), True),
    ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ([0, 1, 2], [0, 1, 2]), True),
    ([[0, 1, 0], [1, 0, 0], [0, 0, 1]], ([0, 1, 2], [1, 0, 2]), True),
    ([[0, 25, 0], [18, 0, 0], [0, 0, 8]], ([0, 1, 2], [1, 0, 2]), True),
    (
        [[0.9, 0.08, 0.02], [0.2, 0.7, 0.1], [0.1, 0.1, 1.8]],
        ([0, 1, 2], [0, 1, 2]),
        True,
    ),
    # Identical preds (i.e. same scores with unique refs)
    ([[1, 0, 1], [0, 1, 0], [0, 0, 1]], ([0, 1, 2], [0, 1, 2]), True),
    ([[0.1, 1, 1], [0, 1, 0], [0, 0, 1]], ([0, 1, 2], [0, 1, 2]), True),
    ([[1, 1, 1], [0, 1, 1], [0, 0, 1]], ([0, 1, 2], [0, 1, 2]), True),
    ([[0, 1, 0], [1, 0, 1], [0, 0, 1]], ([0, 1, 2], [1, 0, 2]), True),
    ([[1, 1, 1], [1, 0, 0], [0, 0, 0]], ([0, 1, 2], [1, 0, 2]), True),
    ([[0, 1, 1], [0, 1, 0], [0, 0, 1]], ([0, 1, 2], [2, 1, 0]), True),  # pred 0 scores
    ([[0, 1, 0], [0, 1, 0], [0, 0, 0]], ([0, 1, 2], [1, 0, 2]), True),  # pred 0 scores
    # Unequal number of refs/preds
    ([[0, 1, 0], [1, 0, 1]], ([0, 1], [1, 0]), True),
    ([[0, 1, 0], [0.9, 0, 1]], ([0, 1], [1, 2]), True),  # 2 refs for 3 preds
    ([[0, 1], [0.9, 0], [1, 0]], ([0, 2], [1, 0]), True),  # 3 refs for 2 preds
    ([[0, 1, 0, 0], [0.9, 0, 1, 0]], ([0, 1], [1, 2]), True),  # 2 refs for 4 preds
    # Cases for which a greedy algorithm would not produce the most optimized matching
    ([[0.9, 0.2, 0.05], [0.8, 0, 0], [0.1, 0.1, 1.8]], ([0, 1, 2], [1, 0, 2]), True),
    # Cases for "lower is better" metrics, the matching minimizes the cost/weights
    ([[1, 0]], ([0], [1]), False),
    ([[1, 0.5, 0], [0, 1, 0], [0, 0.2, 1]], ([0, 1, 2], [2, 0, 1]), False),
]


@pytest.mark.parametrize(("scores", "expected_pairs", "maximize"), TEST_CASES)
def test_alignment_match(
    scores: list[list[int]],
    expected_pairs: tuple[list[int], list[int]],
    maximize: bool,
) -> None:
    indexes_pairs = compute_matching_from_score_matrix(
        np.array(scores, dtype=np.float32), maximize=maximize
    )
    assert (indexes_pairs[0].tolist(), indexes_pairs[1].tolist()) == expected_pairs
