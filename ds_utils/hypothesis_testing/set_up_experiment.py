"""
Helper functions when setting up an experiment.
"""

import numpy as np
from statsmodels.stats import power as stats_power


def _calculate_effect_size_means(baseline_mean: float, new_mean: float, standard_deviation: float) -> float:
    """
    Calculate Cohen's d, a standardised difference between two means i.e. how different the new mean is vs the baseline
    in terms of standard deviations. This is performed because the effect size would depend on the unit of measurement
    otherwise, e.g. £0.50 vs £0.75 has a difference of 0.25, whereas 50 pence vs 75 pence would have a difference of 25
    without standardising.

    More info here: https://en.wikipedia.org/wiki/Effect_size#Cohen's_d

    Parameters
    ----------
    baseline_mean : float
        Baseline value that reflects the current metric we are trying to change e.g. average spend.
    new_mean : float
        The smallest meaningful effect that we wish to be able to detect i.e. at what point do the results become
        commercially interesting e.g. £12.50 average spend vs £10.00 baseline may be the smallest shift which yields a
        financial benefit that makes the project worth implementing.
    standard_deviation : float
        Standard deviation for the metric being tested.

    Returns
    -------
    float
        The effect size, or how different the two means are. Rules of thumb are that...
            Cohen's d = 0.20: small effect size
            Cohen's d = 0.50: medium effect size
            Cohen's d = 0.80: large effect size
    """

    # Calculate h according to the formula:
    # |(new_mean - baseliine)| / standard deviation
    return abs(new_mean - baseline_mean) / standard_deviation


def _calculate_effect_size_proportions(baseline_proportion: float, new_proportion: float) -> float:
    """
    Calculate Cohen's h, a measure of distance between two proportions or probabilities as per
    https://en.wikipedia.org/wiki/Cohen%27s_h.

    Parameters
    ----------
    baseline_proportion : float in interval (0,1)
        Baseline value that reflects the current metric we are trying to change e.g. the existing
        retention rate.
    new_proportion : float in interval (0,1)
        The smallest meaningful effect that we wish to be able to detect i.e. at what point do the results become
        commercially interesting e.g. 85% retention vs 80% baseline may be the smallest shift which yield a financial
        benefit that makes the project worth implementing.

    Returns
    -------
    float
        The effect size, or how different the two prortions are. Rules of thumb are that...
            Cohen's h = 0.20: small effect size
            Cohen's h = 0.50: medium effect size
            Cohen's h = 0.80: large effect size
    """

    # Calculate h according to the formula:
    # |( 2 * arcsin(√new_proportion) ) - ( 2 * arcsin (√baseline_proportion) )|
    sqrt_baseline = np.sqrt(baseline_proportion)
    sqrt_new_probability = np.sqrt(new_proportion)

    double_arcsin_sqrt_baseline = 2 * (np.arcsin(sqrt_baseline))
    double_arcsin_sqrt_new_prob = 2 * (np.arcsin(sqrt_new_probability))

    return abs(double_arcsin_sqrt_new_prob - double_arcsin_sqrt_baseline)


def calculate_required_sample_size(
        baseline_metric_value: float,
        new_metric_value: float,
        measurement_type: str,
        alternative_hypothesis: str = 'two-sided',
        power: float = 0.8,
        significance_level: float = 0.05,
        standard_deviation: float = None,
) -> int:
    """
    Calculate the required sample size for an experiment given a certain shift we are trying to predict.

    Parameters
    ----------
    baseline_metric_value : float
        Baseline value that reflects the current metric we are trying to change e.g. the existing
        retention rate.
    new_metric_value : float
        The smallest meaningful effect that we wish to be able to detect i.e. at what point do the results become
        commercially interesting e.g. 85% retention vs 80% baseline may be the smallest shift which yield a financial
        benefit that makes the project worth implementing.
    measurement_type : str
        Whether the metric is a proportion (e.g. % conversion rate) or mean (e.g. average spend).
    alternative_hypothesis : str (default is 'two-sided')
        Whether you are running a 'two-sided' test, or checking whether the new metric will be 'smaller' or 'larger'.
        'two-sided' is generally recommended because we do not know in advance whether the change in our experiment
        will yield positive or negative results.
    power : float in interval (0,1) (default is 0.8)
        Probability that the test correctly rejects the Null Hypothesis if the Alternative Hypothesis is true
        i.e. likelihood of detecting a shift when it is genuine (one minus the probability of a type II error).
        Default value of 80% is commonly used but you should consider what is appropriate given the business context.
    significance_level : float in interval (0,1) (default is 0.05)
        The significance level/probability of a type I error, i.e. likelihood of a false positive (incorrectly rejecting
        the Null Hypothesis when it is in fact true). Default value of 5% is commonly used but you should consider what
        is appropriate given the business context.
    standard_deviation : float (default is none)
        Standard deviation for the metric being tested. Only needs to be set if `measurement_type` is 'mean'.

    Returns
    -------
    int
        Minimum sample size required to satisfy experiment criteria.

    Raises
    ----------
    TypeError
        If `measurement_type` is 'mean' but not `standard_deviation` provided.
    ValueError
        If `significance_level` or `power` not in range (0,1).
        If `measurement_type` not in ['proportion', 'mean'].
    """

    # Errors in arguments provided
    if measurement_type == 'mean' and standard_deviation is None:
        raise TypeError("When measuring a mean for your test, you must also specify its existing `standard_deviation`.")

    if measurement_type not in ('proportion', 'mean'):
        raise ValueError('measurement_type must be "proportion" or "mean".')

    if not 0 < significance_level < 1:
        raise ValueError("significance_level must be greater than 0 but less than 1.")

    if not 0 < power < 1:
        raise ValueError("power must be greater than 0 but less than 1.")

    # Calculate sample size required if measuring difference between two proportions
    if measurement_type == 'proportion':

        # How big is the shift we want to capture
        effect_size = _calculate_effect_size_proportions(
            baseline_proportion=baseline_metric_value,
            new_proportion=new_metric_value
        )

        # Given our experiment parameters, what sample size is needed
        required_sample_size = stats_power.zt_ind_solve_power(
            effect_size=effect_size,
            alpha=significance_level,
            power=power,
            alternative=alternative_hypothesis,
        )

    # Calculate sample size required if measuring difference between two means
    elif measurement_type == 'mean':

        # How big is the shift we want to capture
        effect_size = _calculate_effect_size_means(
            baseline_mean=baseline_metric_value,
            new_mean=new_metric_value,
            standard_deviation=standard_deviation
        )

        # Given our experiment parameters, what sample size is needed
        required_sample_size = stats_power.tt_ind_solve_power(
            effect_size=effect_size,
            alpha=significance_level,
            power=power,
            alternative=alternative_hypothesis,
        )

    return int(required_sample_size)
