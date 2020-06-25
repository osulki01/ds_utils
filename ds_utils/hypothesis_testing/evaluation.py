"""
Analyse the outcome of an experiment and test for significance.
"""

# Standard library imports
from typing import Tuple

# Third party imports
import numpy as np
from statsmodels.stats import weightstats


def parametric_significance_test_on_raw_scores(
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
    """

    # Invalid significance level
    if not 0 < significance_level < 1:
        raise ValueError("significance_level must be greater than 0 but less than 1.")

    # Perform t-test for means
    if measurement_type == 'mean':

        test_statistic, p_value, _ = weightstats.ttest_ind(
            x1=group_1_observations,
            x2=group_2_observations,
            alternative=alternative_hypothesis
        )

    # Perform z-test for proportions
    elif measurement_type == 'proportion':

        # Check values are encoded as 1 for occurrence of event, and 0 for non-event
        binary_values = {0, 1}

        if set(group_1_observations) != binary_values or set(group_2_observations) != binary_values:
            raise ValueError(
                'When testing proportions, values must be marked as 1 to represent the event, and 0 for non-event'
            )

        test_statistic, p_value = weightstats.ztest(
            x1=group_1_observations,
            x2=group_2_observations,
            alternative=alternative_hypothesis
        )

    else:
        raise ValueError('measurement_type must be "proportion" or "mean".')

    # Display interpretation if requested
    if verbose:
        if p_value < significance_level:
            print('\nSIGNIFICANT DIFFERENCE between the two groups!\n')
            print(f'Test returns a p-value of {p_value:.3f},'
                  f' which means we CAN reject the null at a significance level of {significance_level * 100:.1f}%')
        else:
            print('\nNO SIGNIFICANT DIFFERENCE between the two groups.\n')
            print(f'Test returns a p-value of {p_value:.3f},'
                  f' which means we CANNOT reject the null at a significance level of {significance_level * 100:.1f}%')

    return p_value, test_statistic
