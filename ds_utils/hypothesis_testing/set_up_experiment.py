"""
Helper functions when setting up an experiment.
"""

# Standard library imports
from typing import Dict, Union

# Third party imports
import numpy as np
from numpy import random
import pandas as pd
from statsmodels.stats import power as stats_power


def calculate_required_sample_size(
        baseline_metric_value: float,
        new_metric_value: float,
        measurement_type: str,
        *,
        alternative_hypothesis: str = 'two-sided',
        power: float = 0.8,
        significance_level: float = 0.05,
        standard_deviation: float = None,
) -> int:
    """
    Calculate the required sample size for an experiment given a certain degree of change that we want to confidently
    detect.

    Parameters
    ----------
    baseline_metric_value : float
        Baseline value that reflects the current metric we are trying to change e.g. the existing retention rate.
    new_metric_value : float
        The smallest meaningful effect that we wish to be able to detect i.e. at what point do the results become
        commercially interesting e.g. 85% retention vs 80% baseline may be the smallest shift which yield a financial
        benefit that makes the project worth implementing.
    measurement_type : str (must be 'proportion' or 'mean')
        Whether the metric is a proportion (e.g. % conversion rate) or mean (e.g. average spend).
    alternative_hypothesis : str 'two-sided' (default), 'larger', 'smaller'
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
        If `measurement_type` is 'mean' but no `standard_deviation` provided.
    ValueError
        If `significance_level` or `power` not in range (0,1).
    ValueError
        If `measurement_type` not in ['proportion', 'mean'].
    """

    # Errors in arguments provided
    if measurement_type == 'mean' and standard_deviation is None:
        raise TypeError("When measuring a mean for your test, you must also specify its existing `standard_deviation`.")

    if not 0 < significance_level < 1:
        raise ValueError("significance_level must be greater than 0 but less than 1.")

    if not 0 < power < 1:
        raise ValueError("power must be greater than 0 but less than 1.")

    # Calculate sample size required if measuring difference between two proportions and will therefore use a z-test
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

    # Calculate sample size required if measuring difference between two means and will therefore use a t-test
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

    else:
        raise ValueError('measurement_type must be "proportion" or "mean".')

    return int(required_sample_size)


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


def create_sample_groups(original_population: pd.DataFrame, sample_groups: Union[dict, list]) -> pd.DataFrame:
    """
    Randomly assign records from a population dataset to distinct sample groups.

    Parameters
    ----------
    original_population : pd.DataFrame
        The total dataset from which samples will be drawn.
    sample_groups : dict[str, float] or dict[str, int] or list[str]
        Can be a dictionary with the name of each sample group, and its size expressed as a proportion of the
        population or absolute size in terms of number of records. Or this can be a list of the names of each sample
        group, indicating that the population should be split evenly across them.
    original_population : pd.DataFrame
        The total dataset from which samples will be drawn.
    sample_groups : dict[str, float] or dict[str, int]
        Keys: The names of each sample group
        Values: How big they should be.

    Returns
    -------
    pd.DataFrame
        Copy of the original dataset, but with an additional column called 'sample_group' denoting the group that
        each record has been assigned to, and only containing records that have been assigned (e.g. if we took a sample
        that is smaller than the population).

    Raises
    ------
    ValueError
        If the values provided for the sizes of each sample group are not all floats (proportions), or all integers
        (absolute sizes).
    ValueError
        If the proportions do not adhere to 0 < proportion < 1.
    ValueError
        If the proportions sum up to more than 1.
    ValueError
        If the absolute sizes sum up to more than the size of the original population.
    """

    # Take copy so original data is not accidentally altered, and add a new column which will store the group that
    # observations have been assigned to
    samples_from_population = original_population.copy()
    samples_from_population['sample_group'] = np.nan

    # Randomly shuffle all of the possible row numbers to sample from
    population_size = original_population.shape[0]
    population_indices_shuffled = random.permutation(population_size)

    # If a list of group names is provided, then divide the population evenly across the groups
    if isinstance(sample_groups, list):

        # Split all of the indices evenly, then assign the relevant records to each group
        group_indices = np.array_split(population_indices_shuffled, len(sample_groups))

        for group, indices in zip(sample_groups, group_indices):
            samples_from_population.iloc[indices, -1] = group

    else:
        # Check how user has specified sample sizes
        sample_size_type = _check_if_sample_sizes_are_proportions_or_absolute(sample_groups)
        _check_sample_size_values_are_appropriate(original_population, sample_groups, sample_size_type)

        # Work through the randomly ordered row indices and assign an appropriate size to each sample group
        start_index = 0
        for group in sample_groups:
            # Either take the absolute size, or multiply the proportion by the population size
            group_size = \
                sample_groups[group] if sample_size_type == 'absolute' else int(sample_groups[group] * population_size)

            # Extract the row numbers that will be assigned to this group
            end_index = start_index + group_size
            group_indices = population_indices_shuffled[start_index: end_index]

            # Record their assignment
            samples_from_population.iloc[group_indices, -1] = group

            # Reset to our new starting point
            start_index = end_index

    # Return all of the records that were assigned to a sample group
    return samples_from_population.loc[~samples_from_population['sample_group'].isna(), :]


def _check_if_sample_sizes_are_proportions_or_absolute(sample_groups: Dict[str, Union[float, int]]) -> str:
    """
    Identifies whether the user has specified the size of each sample group as a proportion of the population, or their
    absolute size in terms of number of records.

    Parameters
    ----------
    sample_groups : dict[str, float] or dict[str, int]
        Keys: The names of each sample group
        Values: How big they should be.

    Returns
    -------
    str
        Whether the sizes reflect 'proportion' or their 'absolute' size.

    Raises
    ----------
    ValueError
        If the values provided for the sizes of each sample group are not all floats (proportions), or all integers
        (absolute sizes).
    """

    # Check how the sizes of each sample group have been provided
    user_provided_types = set()

    for group_size in sample_groups.values():
        user_provided_types.add(type(group_size))

    # When taking proportions of the overall population, every value should be a float
    if user_provided_types == {float}:
        return 'proportion'
    # When using specific numbers for each group, every value should be an integer
    elif user_provided_types == {int}:
        return 'absolute'
    else:
        raise ValueError(
            'The sizes provided as values for sample_groups must all be proportions (e.g. 0.8, 0.1, 0.1) or '
            'absolute sizes (e.g. 65, 40, 25).'
        )


def _check_sample_size_values_are_appropriate(
        original_population: pd.DataFrame,
        sample_groups: Dict[str, Union[float, int]],
        size_type: str,
) -> None:
    """
    Checks that the values provided for the size of each sample group, be they proportions or absolute sizes, are valid.

    Parameters
    ----------
    original_population : pd.DataFrame
        The total dataset from which samples will be drawn.
    sample_groups : dict[str, float] or dict[str, int]
        Keys: The names of each sample group
        Values: How big they should be.
    size_type : str 'proportion', 'absolute'
        Whether the sizes reflect 'proportion' or their 'absolute' size.

    Raises
    -------
    ValueError
        If the proportions do not adhere to 0 < proportion < 1.
    ValueError
        If the proportions sum up to more than 1.
    ValueError
        If the absolute sizes sum up to more than the size of the original population.
    """

    if size_type == 'proportion':

        for proportion_size in sample_groups.values():
            if not 0 < proportion_size < 1:
                raise ValueError("Proportions should all be 0 < proportion < 1.")

        if sum(sample_groups.values()) > 1:
            raise ValueError("Proportions should not exceed 1 in total.")

    elif size_type == 'absolute':

        if sum(sample_groups.values()) > original_population.shape[0]:
            raise ValueError(
                "The sum of all sample sizes should not exceed that of the original population that you are sampling."
            )
