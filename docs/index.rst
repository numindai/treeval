.. treeval documentation master file, created by
   sphinx-quickstart on Tue Nov 12 11:37:41 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Treeval documentation
======================

Welcome to Treeval's documentation.

Treeval is a Python package providing an easy and flexible way to evaluate the matching of tree-based data, i.e. dictionaries. It is initially developed to evaluate the results of `structured data extraction tasks <https://numind.ai/blog/nuextract-a-foundation-model-for-structured-extraction>`_ from language models, but can be used with any type of leaf values and metrics.

This documentation introduces the way Treeval works, the metrics and how to use your owns.

Getting started
-----------------------------

Treeval can be installed from PyPi by running:

..  code-block:: bash

    pip install miditok

Contents
-----------------------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   conceptual_guide
   treeval
   metrics
   the_treeval_score
   examples

.. toctree::
   :hidden:
   :caption: Project Links

   GitHub <https://github.com/NuMindAI/Treeval>
   PyPi <https://pypi.org/project/treeval/>
