
===================================
Code examples
===================================

This pages show Treeval usage examples.

Creating a schema and tree metrics
----------------------------------

The ``schema`` is the description of the tree. It is a dictionary mapping node names (keys) to leaf types (string values). The schema is used by the :py:func:`treeval.treeval` method to efficiently parse trees and batch metrics computations. It is used in combination with a ``tree_metrics``, which is dictionary with the same structure (node names/keys) as the schema where leaves values are sets of names of the metrics to compute scores on the associated leaves.

..  code-block:: python

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

Aggregating results
-----------------------------

:ref:`Treeval method`
