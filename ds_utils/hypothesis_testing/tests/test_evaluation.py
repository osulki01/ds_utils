"""
Testing for analysing the outcome of an experiment and testing for significance.
"""

# Third party imports
import numpy as np
from statsmodels.stats import weightstats

# Local application imports
from ds_utils.hypothesis_testing import evaluation


def test_parametric_significance_test_on_raw_observations_mean():
    """
    Ensure parametric experiment (when working with the raw observations) is evaluated correctly when the experiment
    metric is a mean.
    """

    group_1_observations = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1])
    group_2_observations = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    alternative_hypothesis = 'two-sided'

    expected_test_statistic, expected_p_value, _ = weightstats.ttest_ind(
        x1=group_1_observations,
        x2=group_2_observations,
        alternative=alternative_hypothesis
        )

    actual_p_value, actual_test_statistic = evaluation.parametric_significance_test_on_raw_observations(
        group_1_observations=group_1_observations,
        group_2_observations=group_2_observations,
        alternative_hypothesis=alternative_hypothesis,
        measurement_type='mean'
    )

    assert actual_test_statistic == expected_test_statistic
    assert actual_p_value == expected_p_value


def test_parametric_significance_test_on_raw_observations_proportion():
    """
    Ensure parametric experiment (when working with the raw observations) is evaluated correctly when the experiment
    metric is a proportion.
    """

    group_1_observations = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0])
    group_2_observations = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0])
    alternative_hypothesis = 'two-sided'

    expected_test_statistic, expected_p_value = weightstats.ztest(
        x1=group_1_observations,
        x2=group_2_observations,
        alternative=alternative_hypothesis
        )

    actual_p_value, actual_test_statistic = evaluation.parametric_significance_test_on_raw_observations(
        group_1_observations=group_1_observations,
        group_2_observations=group_2_observations,
        alternative_hypothesis=alternative_hypothesis,
        measurement_type='proportion'
    )

    assert actual_test_statistic == expected_test_statistic
    assert actual_p_value == expected_p_value


def test__print_interpretation_of_p_value(capsys):
    """Correct message should be displayed to user depending on whether the results are significant or not."""

    p_value_lower = 0.04
    significance_level = 0.05
    p_value_higher = 0.06

    # ---------------------------------------------------------------------------------------
    # SCENARIO 1
    # Capture what the function prints when the p-value is LOWER than the significance_level
    # ---------------------------------------------------------------------------------------

    evaluation._print_interpretation_of_p_value(p_value_lower, significance_level)
    actual_interpretation_when_significant, _ = capsys.readouterr()

    expected_interpretation_when_significant = (
        f'\nSIGNIFICANT DIFFERENCE between the two groups!\n'
        f'Test returns a p-value of {p_value_lower:.3f}, which means we CAN reject the null at a significance level of '
        f'{significance_level * 100:.1f}%'
    )

    # Only evaluate the content, not necessarily how the linebreaks are formatted
    actual_interpretation_when_significant = actual_interpretation_when_significant.replace('\n', '')
    expected_interpretation_when_significant = expected_interpretation_when_significant.replace('\n', '')

    assert actual_interpretation_when_significant == expected_interpretation_when_significant

    # ---------------------------------------------------------------------------------------
    # SCENARIO 2
    # Capture what the function prints when the p-value is HIGHER than the significance_level
    # ---------------------------------------------------------------------------------------

    evaluation._print_interpretation_of_p_value(p_value_higher, significance_level)
    actual_interpretation_when_not_significant, _ = capsys.readouterr()

    expected_interpretation_when_not_significant = (
        f'\nNO SIGNIFICANT DIFFERENCE between the two groups.\n'
        f'Test returns a p-value of {p_value_higher:.3f}, which means we CANNOT reject the null at a significance level'
        f'of {significance_level * 100:.1f}%'
    )

    # Only evaluate the content, not necessarily how the linebreaks are formatted
    actual_interpretation_when_not_significant = actual_interpretation_when_not_significant.replace('\n', '')
    expected_interpretation_when_not_significant = expected_interpretation_when_not_significant.replace('\n', '')

    assert actual_interpretation_when_not_significant == expected_interpretation_when_not_significant
