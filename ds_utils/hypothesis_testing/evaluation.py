"""
Analyse the outcome of an experiment and test for significance.
"""

# Standard library imports
from typing import Tuple

# Third party imports
import numpy as np
from statsmodels.stats import weightstats

# Local application imports
from ds_utils.hypothesis_testing import _check_experiment_inputs


def parametric_significance_test_on_raw_observations(
        group_1_observations: np.ndarray,
        group_2_observations: np.ndarray,
        measurement_type: str,
        alternative_hypothesis: str = 'two-sided',
        significance_level: float = 0.05,
        verbose: bool = True,
) -> Tuple[float, float]:
    """
    Tests for a significant difference between the observations recorded for two experimental groups.

    Parameters
    ----------
    group_1_observations : numpy array_like
        Observations for specific group in the experiment.
    group_2_observations : numpy array_like
        Observations for other group in the experiment which group_1 will be compared against.
    measurement_type : str 'proportion', 'mean'
        Whether the metric is a proportion (e.g. % conversion rate) or mean (e.g. average spend).
    alternative_hypothesis : str 'two-sided' (default), 'larger', 'smaller'
        Whether you are running a 'two-sided' test, or checking whether the new metric will be 'smaller' or 'larger'.
        'two-sided' is generally recommended because we do not know in advance whether the change in our experiment
        will yield positive or negative results.
    significance_level : float in interval (0,1) (default is 0.05)
        The significance level/probability of a type I error, i.e. likelihood of a false positive (incorrectly rejecting
        the Null Hypothesis when it is in fact true). Default value of 5% is commonly used but you should consider what
        is appropriate given the business context.
    verbose : bool

    Returns
    -------
    Tuple[float, float]
        (1st value) p-value, i.e. the probability of obtaining results as extreme as the observed result.
        (2nd value) test-statistic, which applies to z-test when measuring proportions, and t-test for means.

    Raises
    ------
    ValueError
        If `significance_level` does not adhere to 0 < significance_level < 1.
    ValueError
        If the experiment metric is a proportion, but the individual observations are not all represented as 0 or 1.
    """

    # Validate parameters of the experiment are appropriate
    _check_experiment_inputs.validate_experiment_parameter_between_0_and_1(significance_level, 'significance_level')
    _check_experiment_inputs.validate_measurement_type_is_valid(measurement_type)

    # Perform t-test for means
    if measurement_type == 'mean':

        test_statistic, p_value, _ = weightstats.ttest_ind(
            x1=group_1_observations,
            x2=group_2_observations,
            alternative=alternative_hypothesis
        )

    # Perform z-test for proportions
    elif measurement_type == 'proportion':

        for observations in [group_1_observations, group_2_observations]:
            _check_experiment_inputs.validate_binary_events_are_represented_with_0_or_1(observations)

        test_statistic, p_value = weightstats.ztest(
            x1=group_1_observations,
            x2=group_2_observations,
            alternative=alternative_hypothesis
        )

    # Display interpretation if requested
    if verbose:
        _print_interpretation_of_p_value(p_value, significance_level)

    return p_value, test_statistic


def _print_interpretation_of_p_value(p_value: float, significance_level: float) -> None:
    """
    Prints message for the user indicating whether the differences observed in the experiment can be deemed significant.

    Parameters
    ----------
    p_value : float
        Assuming the null hypothesis is true, the probability of observing a result as extreme or more extreme
        than the one actually recorded.
    significance_level : float in interval (0,1)
        The significance level/probability of a type I error for the experiment.
    """

    if p_value < significance_level:
        print(
            f'\nSIGNIFICANT DIFFERENCE between the two groups!\n'
            f'Test returns a p-value of {p_value:.3f}, which means we CAN reject the null at a significance level of '
            f'{significance_level * 100:.1f}%'
        )
    else:
        print(
            f'\nNO SIGNIFICANT DIFFERENCE between the two groups.\n'
            f'Test returns a p-value of {p_value:.3f}, which means we CANNOT reject the null at a significance level'
            f'of {significance_level * 100:.1f}%'
        )
