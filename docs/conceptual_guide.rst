.. _conceptual-guide-label:

====================================
Conceptual guide of evaluating trees
====================================

This page details how treeval works conceptually.

Evaluating tree-based data
-----------------------------

Evaluating tree-based data, i.e. computing metrics between reference and hypothesis trees, can be tricky as the samples can feature complex structure and various types of data.

Recursively parses the two predicted and reference dictionaries, and run metrics on each common **leaves**.

Tree structure covered by treeval
---------------------------------

Dict/lists

Everything else (other types) are treated as leaves.

Recursively parses the two predicted and reference dictionaries, and run metrics on each common **leaves**.
Lists --> similarity matrix --> assignment problem

Evaluation of lists of items
-----------------------------

When a leaf is a list of objects (that may be of any type, including dictionaries hence child trees), treeval does not consider the positions of elements within a reference and a prediction. Indeed, in many cases, a list within a tree can be constructed in various orders, hence evaluating the elements pairwise can be irrelevant. Additionally, even when the construction of the list is supposed to follow a strict order, the predicted list might miss a few expected elements or contain unexpected elements, at any position, thus breaking a potential pairwise evaluation. Consequently, the evaluation must be **permutation-invariant**.

Treeval handles this requirement by computing the metrics scores on all the possible combinations of pairs of reference/prediction items within the lists, and keeps the maximum score met for each reference item.

`assignment problem <https://en.wikipedia.org/wiki/Assignment_problem>`_

In our case, leafs can be evaluated with several metrics, that must all be considered when
Treeval consider all metrics equally, weighted equally (a potential evolution could allow to weight metrics differently in the assignment process)

List of dicts --> recursively evaluate the subtrees
