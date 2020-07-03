"""
Testing for helper functions when setting up an experiment.
"""

# Standard library imports
import math

# Third party imports
import numpy as np
import pandas as pd
import pytest
from statsmodels.stats import power as stats_power

# Local application imports
from ds_utils.hypothesis_testing import set_up_experiment


def test__calculate_effect_size_means():
    """
    Value is calculated using Cohen's d formula
    |Mean_1 - Mean_2| / standard_deviation
    as per https://en.wikipedia.org/wiki/Effect_size#Cohen's_d.
    """

    baseline_mean = 9.5
    new_mean = 10
    standard_deviation = 0.25

    expected_effect_size = math.fabs(new_mean - baseline_mean) / standard_deviation

    actual_effect_size = set_up_experiment._calculate_effect_size_means(
        baseline_mean=baseline_mean,
        new_mean=new_mean,
        standard_deviation=standard_deviation
    )

    assert actual_effect_size == pytest.approx(expected_effect_size, abs=1e-3)


def test__calculate_effect_size_proportions():
    """
    Value is calculated using Cohen's h formula
    |( 2 * arcsin(√new_proportion) ) - ( 2 * arcsin (√baseline_proportion) )|
    as per https://en.wikipedia.org/wiki/Cohen%27s_h.
    """

    baseline_proportion = 0.5
    new_proportion = 0.4

    transformed_baseline_proportion = 2 * math.acos(math.sqrt(baseline_proportion))
    transformed_new_proportion = 2 * math.acos(math.sqrt(new_proportion))

    expected_effect_size = math.fabs(transformed_new_proportion - transformed_baseline_proportion)

    actual_effect_size = set_up_experiment._calculate_effect_size_proportions(
        baseline_proportion=baseline_proportion,
        new_proportion=new_proportion
    )

    assert actual_effect_size == pytest.approx(expected_effect_size, abs=1e-3)


@pytest.fixture
def global_variables_for_calculate_required_sample_size():
    """Values which can be re-used across the tests."""
    pytest.alternative_hypothesis = 'two-sided'
    pytest.baseline_metric_value = 0.75
    pytest.new_metric_value = 0.78
    pytest.power = 0.8
    pytest.significance_level = 0.05
    pytest.standard_deviation = 0.1


def test_calculate_required_sample_size_mean(global_variables_for_calculate_required_sample_size):
    """
    Sample size should be calculated appropriately using the statsmodels package when measuring a mean value in the
    experiment (using a t-test).
    """

    effect_size = set_up_experiment._calculate_effect_size_means(
        baseline_mean=pytest.baseline_metric_value,
        new_mean=pytest.new_metric_value,
        standard_deviation=pytest.standard_deviation
    )

    expected_required_sample_size = stats_power.tt_ind_solve_power(
        effect_size=effect_size,
        alpha=pytest.significance_level,
        power=pytest.power,
        alternative=pytest.alternative_hypothesis,
        )

    actual_required_sample_size = set_up_experiment.calculate_required_sample_size(
        baseline_metric_value=pytest.baseline_metric_value,
        new_metric_value=pytest.new_metric_value,
        measurement_type='mean',
        alternative_hypothesis=pytest.alternative_hypothesis,
        power=pytest.power,
        significance_level=pytest.significance_level,
        standard_deviation=pytest.standard_deviation,
    )

    assert actual_required_sample_size == int(expected_required_sample_size)


def test_calculate_required_sample_size_proportion(global_variables_for_calculate_required_sample_size):
    """
    Sample size should be calculated appropriately using the statsmodels package when measuring a proportion value in
    the experiment (using a z-test).
    """

    effect_size = set_up_experiment._calculate_effect_size_proportions(
        baseline_proportion=pytest.baseline_metric_value,
        new_proportion=pytest.new_metric_value,
    )

    expected_required_sample_size = stats_power.zt_ind_solve_power(
        effect_size=effect_size,
        alpha=pytest.significance_level,
        power=pytest.power,
        alternative='two-sided',
        )

    actual_required_sample_size = set_up_experiment.calculate_required_sample_size(
        baseline_metric_value=pytest.baseline_metric_value,
        new_metric_value=pytest.new_metric_value,
        measurement_type='proportion',
        alternative_hypothesis=pytest.alternative_hypothesis,
        power=pytest.power,
        significance_level=pytest.significance_level,
    )

    assert actual_required_sample_size == int(expected_required_sample_size)


def test_calculate_required_sample_size_raises_exception_if_means_provided_without_standard_deviation(
        global_variables_for_calculate_required_sample_size
):
    """
    The standard deviation is required to calculated the standardised effect size between two means, so the function
    should not run if no standard deviation is given when that is the measurement type.
    """

    with pytest.raises(
            TypeError,
            match="When measuring a mean for your test, you must also specify its existing `standard_deviation`."
    ):
        set_up_experiment.calculate_required_sample_size(
            baseline_metric_value=pytest.baseline_metric_value,
            new_metric_value=pytest.new_metric_value,
            measurement_type='mean',
        )


def test_create_sample_groups_correct_absolute_sizes():
    """The number of records assigned to each group should match the absolute sizes specified by the user."""

    # Show how many rows should belong to each sample group
    expected_sample_sizes = pd.DataFrame(
        index=['Group_1', 'Group_2', 'Group_3'],
        columns=['sample_group_size'],
        data=[4, 4, 2],
    )
    expected_sample_sizes.index.set_names('sample_group', inplace=True)

    # Create population and assign records to sample groups
    original_population_10_rows = pd.DataFrame(np.arange(10), columns=['original_row_index'])

    sample_groups = set_up_experiment.create_sample_groups(
        original_population=original_population_10_rows,
        sample_groups={'Group_1': 4, 'Group_2': 4, 'Group_3': 2},
    )

    # Check that the sizes align
    actual_sample_sizes = sample_groups \
        .groupby('sample_group')\
        .agg(sample_group_size=pd.NamedAgg(column='sample_group', aggfunc='size'))

    pd.testing.assert_frame_equal(actual_sample_sizes, expected_sample_sizes)


def test_create_sample_groups_correct_even_split_sizes():
    """If a list of sample group names are provided only, then the population should be split evenly across them."""

    # Show how many rows should belong to each sample group
    expected_sample_sizes = pd.DataFrame(
        index=['Group_1', 'Group_2', 'Group_3'],
        columns=['sample_group_size'],
        data=[3, 3, 3],
    )
    expected_sample_sizes.index.set_names('sample_group', inplace=True)

    # Create population and assign records to sample groups
    original_population_9_rows = pd.DataFrame(np.arange(9), columns=['original_row_index'])

    sample_groups = set_up_experiment.create_sample_groups(
        original_population=original_population_9_rows,
        sample_groups=['Group_1', 'Group_2', 'Group_3'],
    )

    # Check that the sizes align
    actual_sample_sizes = sample_groups \
        .groupby('sample_group')\
        .agg(sample_group_size=pd.NamedAgg(column='sample_group', aggfunc='size'))

    pd.testing.assert_frame_equal(actual_sample_sizes, expected_sample_sizes)


def test_create_sample_groups_correct_proportion_sizes():
    """The number of records assigned to each group should match the proportions specified by the user."""

    # Show how many rows should belong to each sample group
    expected_sample_sizes = pd.DataFrame(
        index=['Group_1', 'Group_2', 'Group_3'],
        columns=['sample_group_size'],
        data=[4, 4, 2],
    )
    expected_sample_sizes.index.set_names('sample_group', inplace=True)

    # Create population and assign records to sample groups
    original_population_10_rows = pd.DataFrame(np.arange(10), columns=['original_row_index'])

    sample_groups = set_up_experiment.create_sample_groups(
        original_population=original_population_10_rows,
        sample_groups={'Group_1': 0.4, 'Group_2': 0.4, 'Group_3': 0.2},
    )

    # Check that the sizes align
    actual_sample_sizes = sample_groups \
        .groupby('sample_group')\
        .agg(sample_group_size=pd.NamedAgg(column='sample_group', aggfunc='size'))

    pd.testing.assert_frame_equal(actual_sample_sizes, expected_sample_sizes)
