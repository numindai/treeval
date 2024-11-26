
====================================
Conceptual guide of evaluating trees
====================================

This page details how treeval works conceptually.

Evaluating the leaf values of tree-based data
---------------------------------------------

Evaluating tree-based data, i.e. computing metrics between reference and hypothesis trees, can be tricky as the samples can feature complex structure and various types of data. No metric can reliably measure the similarity between two trees globally, and the popular metrics, for example measuring textual similarity, cannot perform accurately on whole trees as the structure syntax must be considered and children nodes must be evaluated in a permutation invariant way.

Treeval offers a flexible way to evaluate trees by working at the **leaf-level**, by computing several metrics between pairs of reference and hypothesis leaves. Doing so, **treeval computes a tree identical to the reference one with leaf values corresponding to the metrics results between reference and hypothesis leaves**. This offers a way to easily analyze results and interpret them at the leaf-level, but also to aggregate all of them to compute metrics averages, that can optionally be wrapped per leaf type.

.. figure:: resources/tree_schema.svg
   :scale: 60 %
   :alt: Tree schema
   :align: center

   Two superposed trees being evaluated. Superposed nodes are assumed to have the same key not displayed here for ease of read. Circles are non-leaf nodes (dictionaries), other shapes are leaves of different types with diamonds being integers, rectangles being strings, octagons being booleans and rounded-sides rectangles being lists of integers. Some branches (including nodes and/or leaves) can mismatch, such as two integer leaves in this figure.

Treeval works with `dictionaries <https://en.wikipedia.org/wiki/Associative_array>`_, i.e. each node is identified by a unique key, and a value that can be a leaf or another dictionary (i.e. node with children). It is possible to use it with binary search trees, or any type of trees, by converting the nodes to Python dictionaries.

Tree structure and leaf types
---------------------------------

Treeval can work with any type of leaf. These can be the builtin Python types (integers, string...) or custom classes. When parsing a tree, Treeval follows a simple a rule: **all nodes are identified as leaves except if their types are dictionaries**. Doing so, nested dictionaries are recursively parsed until all their leaves have been explored. Below is an code example of a **schema description** of a tree, where the values describe the leaf types or child branch structure.

..  code-block:: python

    tree_schema = {
        "song_name": "string",
        "artist_name": "string",
        "song_duration_in_seconds": "integer",
        "has_lyrics": "boolean",
        "information": {
            "tempo": "integer",
            "time_signature": ["4/4", "4/2", "2/2"],  # one of the element within the list
            "key_signature": "string",
        },
        "instruments": ["string"],  # list of items of type "string"
    }

This example can be filled with data such as:

..  code-block:: python

    structured_data = {
        "song_name": "Wake Me Up Before You Go-Go",
        "artist_name": "Wham!",
        "song_duration_in_seconds": 231,
        "has_lyrics": True,
        "information": {
            "tempo": 81,
            "time_signature": "4/4",
            "key_signature": "C major",
        },
        "instruments": ["synthesizer", "bass guitar", "electric guitar", "drums"],
    }

Treeval computes metrics on leaves, between pairs of references and hypothesis trees. These computation can be **batched** when parsing multiple pairs of trees following the same schema, allowing to benefit from a faster runtime which can be crucial for deep-learning based metrics such as sequence similarity or LLM-as-a-judge methods. Additionally, treeval measures the **precision, recall and F1 scores at the tree-level** of the nodes in the hypothesis trees. Treeval has a specific way to handle lists which we describe in the next section.

Evaluation of lists of items
-----------------------------

When a leaf is a list of objects (that may be of any type, including dictionaries representing child branches), treeval does not consider the order of the elements within a reference and a hypothesis lists. In many cases, a list within a tree can be constructed in various orders, hence evaluating the elements pairwise can be irrelevant. Additionally, even when the construction of the list is supposed to follow a strict order, the predicted list might miss a few expected elements or contain unexpected elements, at any position, thus breaking a potential pairwise evaluation. Consequently, Treeval employs a **permutation-invariant** method.

Treeval computes the metrics scores on all the possible combinations of pairs of reference/hypothesis items within the lists, then computes the pairs of reference/hypothesis items that maximize the sum of all of their scores. This corresponds to a form of `assignment problem <https://en.wikipedia.org/wiki/Assignment_problem>`_ which can be solved by computing the maximum/minimum matching of a weighted bipartite graph. However, a leaf can be evaluated with several metrics which must all be considered when solving this assignment problem. These metrics can have different ranges of possible score values and different directions (lower or higher is better). Before computing the maximum matching, Treeval normalizes all metrics scores between 0 and 1, inverse "lower is better" scores and averages them. Doing so, all metrics are weighted equally. This procedure might however ignore the distributions of scores of individual metrics, thus resulting in metrics that might weight more or less than others in practice. If that is the case, you can "trick" the score normalization by storing different ``score_range`` bounds for specific metrics. More details can be read in :ref:`Metrics in Treeval`.

Finally, lists of dictionaries are evaluated with the same alignment method, except that **pairs of dictionaries are evaluated recursively** and that the leaves metrics scores are aggregated per metric before the normalization step.

Treeval uses scipy's `linear_sum_assignment <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html>`_ method, based on the Hungarian algorithm, to compute the matching, as its `runtime is on par with the best performing implementations <https://github.com/berhane/LAP-solvers?tab=readme-ov-file#output>`_ and that the library is popular and well-maintained.

Precision, Recall, F1 and mismatching tree branches
-------------------------------------------------------------------

When evaluating a pair of reference and hypothesis trees, they might not follow the exact same tree structure, i.e. the hypothesis may have additional nodes and branches that does not exist in the reference (false positives), and/or might miss nodes and branches present in the reference (false negatives).
Additionally some leaves types might be "Null" (Python ``None``) marking an explicit absence of value. All the metrics cannot reliably evaluates cases where one or both values are "Null", or simply absent from the tree, thus computing scores for these cases is tricky. Assigning a "default" penalizing value is another option, that might however "corrupt" the final average metrics scores depending on the proportion of such cases, making difficult to interpret the results and report the performances on the actual "correct" nodes.

For these reasons, **Treeval only computes metrics scores on the pairs of leaves that are both present in the reference and hypothesis**, and **report separately precision, recall and f1 scores at the tree-level of the present of the nodes and leaves**. These results are mapped in the the output :py:func:`treeval.treeval` method by the ``precision_node``, ``recall_node``, ``f1_node``, ``precision_leaf``, ``recall_leaf`` and ``f1_leaf`` keys. The figure below gives a visual representation of how these cases are identified to compute the precision and recall scores.

.. figure:: resources/prf_node_leaf.svg
   :scale: 60 %
   :alt: Precision, Recall and F1 schema
   :align: center

   Figure illustrating mismatches between evaluated trees, the blue one being the reference and the green one the hypothesis. The total number of reference nodes is 7, the root node not being counted. The hypothesis tree possesses 6 nodes following the same structure, that could be programmatically interpreted as having the same dictionary keys, and two additional nodes that does not exist in the reference tree. This results in a ``5/6 = 0.833`` **node precision** and ``5/7 = 0.714`` **node recall**.
   Among the leaves present in both the reference and hypothesis, the hypothesis tree possesses one correctly predicted leaf type (true positive), one correctly predicted ``Null`` leaf (true negative), one mispredicted leaf expected to be ``Null`` (false positive) and no ``Null`` leaf expected to be non-``Null`` (false negative), resulting in a ``1/2 = 0.5`` **leaf precision** and ``1/1 = 1`` **leaf recall**.

Note that the leaf precision and recall scores are only computed from the leaves that are present in both the hypothesis and the reference, i.e. the leaves among the true positive nodes of the nodes precision/recall computation.
The F1 scores are computed from the precision and recall values following the formula: ``f1 = 2 * precision * recall / (precision + recall)``.

Separating these results allows an easier interpretability and makes sure that:

* **metrics results** represents the performances on the leaves that are actually present/well predicted in the hypothesis trees only;
* **node precision/recall/f1** represents the tree structure similarity, which must be validated first before analyzing metrics results;
* **leaf precision/recall/f1** represents the similarity of the presence or absence of result ("Null") in the leaves values.
