.. hypothesis_testing:

ds_utils.hypothesis_testing
===========================

Setting up and evaluating experiments.

.. note::
   Much of the underlying functionality in this module is handled by the excellent
   `statsmodels <https://www.statsmodels.org/stable/index.html>`_ package.


Contents
~~~~~~~~

.. contents::
   :local:
   :depth: 1


How Big Should My Sample Be?
----------------------------

When designing an experiment, you will want to ensure you have a statistically significant sample. The online fashion
retailer Stitch Fix has written a terrific
`article <https://multithreaded.stitchfix.com/blog/2015/05/26/significant-sample/>`_ on the aspects that you should
consider.

:py:func:`ds_utils.hypothesis_testing.set_up_experiment.calculate_required_sample_size` is designed to help you express
these considerations and gain an indication of how big your sample size should be.

Firstly, you must identify what metric you are trying to change, and what a meaningful shift would represent. Often
this involves liaising with Commercial Finance colleagues or those who know the domain best, to attach a £ value to
changes in the metric and understand at what point the shift becomes worth pursuing.

Once you have gone through this process, you know what the values for :py:data:`baseline_metric_value`,
:py:data:`new_metric_value`, and :py:data:`measurement_type` should be. The :py:data:`measurement_type` is important
because whether you are measuring a mean or a percentage dictates what type of testing you will be using.

.. note::
   This method makes calculations on the proviso that you will be using a z-test when measuring proportions, and a
   t-test when measuring means. This involves the assumption that you will have at least 10 records that display the
   positive and negative binary classes when it comes to a proportion, and that the data is normally distributed and we
   do not know the true population standard deviation when it comes to means.

   This `flow-chart <https://bloomingtontutors.com/blog/when-to-use-the-z-test-versus-t-test>`_ articulates the reasons
   why particularly well.

For instance, you may have a web journey with a 50% conversion rate, and an increase to 55% would make it worthwhile
implementing the change you are testing.

.. code-block:: python

   >>> from ds_utils.hypothesis_testing import set_up_experiment

   >>> suggested_sample_size = set_up_experiment.calculate_required_sample_size(
   ...     baseline_metric_value=0.5,
   ...     new_metric_value=0.55,
   ...     measurement_type='proportion,
   ... )

   >>> print(suggested_sample_size)
   1564


In cases where you are measuring a mean, such as average spend per user, you will also need to know the standard
deviation in order to calculate your sample size. In this scenario, we know that users typically spend £10.25. If we
can increase that to £10.75 then we would be happy with the outcome, and we know from observing spend in recent periods
that the average spend has a standard deviation of £5.55.

.. code-block:: python
   :emphasize-lines: 7

   >>> from ds_utils.hypothesis_testing import set_up_experiment

   >>> suggested_sample_size = set_up_experiment.calculate_required_sample_size(
   ...     baseline_metric_value=10.25,
   ...     new_metric_value=10.75,
   ...     measurement_type='mean',
   ...     standard_deviation=5.55,
   ... )

   >>> print(suggested_sample_size)
   1935


Depending on your experiment, you can also specify additional parameters:

* Whether your test is `one-tailed/two-tailed <https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faq-what-are-the-differences-between-one-tailed-and-two-tailed-tests/>`_.
  The default for :py:data:`alternative_hypothesis` is 'two-sided' because we're generally open to learning if the
  metric worsens, not only if it improves.
* The `statistical power <https://www.statisticsteacher.org/2017/09/15/what-is-power/>`_ i.e. the probability of
  detecting a change if it does indeed exist. The default value for :py:data:`power` is 0.8 but you should consider the
  context of your experiment i.e. if there is a big risk associated with failing to detect a genuine effect, then you
  may want to increase your power.
* The `significance level <https://en.wikipedia.org/wiki/Statistical_significance>`_ i.e. the false positive rate. Once
  again, consider your context, and if there is a big risk associated with determining that a effect exists when it does
  not, then you may want to reduce your :py:data:`significance_level` from the default of 5%.

.. code-block:: python
   :emphasize-lines: 7-9

   >>> from ds_utils.hypothesis_testing import set_up_experiment

   >>> suggested_sample_size = set_up_experiment.calculate_required_sample_size(
   ...     baseline_metric_value=0.5,
   ...     new_metric_value=0.55,
   ...     measurement_type='proportion',
   ...     alternative_hypothesis='larger',
   ...     power=0.9,
   ...     significance_level=0.01,
   ... )

   >>> print(suggested_sample_size)
   2594


Creating Your Experimental Groups
---------------------------------

Once you know how big your sample sizes should be, you will need to create individual groups that will make up the
experiment.

The function :py:func:`ds_utils.hypothesis_testing.set_up_experiment.create_sample_groups` allows you to randomly
assign records from a dataframe to distinct experimental groups.

Let's assume we have a group of people where we know their eye colour; this won't be used to separate them into sample
groups, but we may want to retain that information for the analysis, or checking that our sample groups have a similarly
representative distribution.

This is our starting dataset, which will have an additional column appended to the end showing the experimental group
that each row will belong to.

.. code-block:: python

   >>> from ds_utils.hypothesis_testing import set_up_experiment
   >>> import pandas as pd

   >>> population_df = pd.DataFrame(columns=['user_id', 'eye_colour'],
   ...                              data=[[86, 'blue'],
   ...                                    [54, 'brown'],
   ...                                    [31, 'green'],
   ...                                    [95, 'hazel']])

   >>> population_df

      user_id   eye_colour
   0       86         blue
   1       54        brown
   2       31        green
   3       95        hazel


You can evenly split your dataframe by simply providing a list of names for each of your sample groups.

.. code-block:: python
   :emphasize-lines: 3

   >>> sample_groups_df = set_up_experiment.create_sample_groups(
   >>>     original_population=population_df,
   >>>     sample_groups=['Group 1', 'Group 2'],
   >>> )

   >>> sample_groups_df

         user_id   eye_colour   sample_group
   0       86            blue        Group 1
   1       54           brown        Group 2
   2       31           green        Group 2
   3       95           hazel        Group 1


Alternatively, you can specify the number of records you want to assign to each group. If your sizes do not cover the
entire population, then the method will only return the records which were assigned to a sample group.

.. code-block:: python
   :emphasize-lines: 3

   >>> sample_groups_df = set_up_experiment.create_sample_groups(
   >>>     original_population=population_df,
   >>>     sample_groups={'Group 1': 2, 'Group 2': 1},
   >>> )

   >>> sample_groups_df

         user_id   eye_colour   sample_group
   0       86            blue        Group 1
   2       31           green        Group 2
   3       95           hazel        Group 1


Or finally, you can specify the proportion of rows that should be assigned to each group. Once again, if your
proportions do not cover the entire population, then the method will only return the records which were assigned to a
sample group.

.. code-block:: python
   :emphasize-lines: 3

   >>> sample_groups_df = set_up_experiment.create_sample_groups(
   >>>     original_population=population_df,
   >>>     sample_groups={'Group 1': 0.5, 'Group 2': 0.25, 'Group 3': 0.25},
   >>> )

   >>> sample_groups_df

         user_id   eye_colour   sample_group
   0       86            blue        Group 2
   1       54           brown        Group 3
   2       31           green        Group 1
   3       95           hazel        Group 1


Testing For Significance
----------------------------

When your experiment has run and you have some results to analyse, use
:py:mod:`ds_utils.hypothesis_testing.evaluation` to test for significance.


Module Overview
---------------

.. autosummary::

   ds_utils.hypothesis_testing.set_up_experiment


Submodules
----------

set_up_experiment
^^^^^^^^^^^^^^^^^

.. automodule:: ds_utils.hypothesis_testing.set_up_experiment
   :members:
