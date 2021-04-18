.. ds_utils documentation master file, originally created by
   sphinx-quickstart on Sun May 24 13:55:25 2020.


Welcome to ds_utils's documentation!
=====================================

ds_utils is a collection of utility functions designed to make data science tasks easier and fill in some small gaps
that existing libraries do not cover.

It is not currently an installable package, but the plan is to make it so.


Use Cases
=========

.. toctree::
   :maxdepth: 1

   Hypothesis Testing <modules/hypothesis_testing>
   Scikit-learn Utilities <modules/sklearn_utils>


.. installation:

Installation
============

You will need Python 3 available on your machine, which can be installed `here <https://www.python.org/downloads/>`_.

To create a python environment with the necessary libraries you must then install `pipenv <https://pypi.org/project/pipenv/>`_.

Finally, install the libraries within a self-contained environment with the command:

.. code-block:: console

   >>> pipenv sync

Modules
=======

.. autosummary::

   ds_utils.hypothesis_testing
   ds_utils.sklearn_utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
