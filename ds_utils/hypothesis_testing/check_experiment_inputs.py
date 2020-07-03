"""
Internal checks within the package to ensure that the parameters provided when setting up or evaluating an experiment are
valid.
"""

# Standard library imports
from typing import Dict, Union

# Third party imports
import numpy as np
import pandas as pd


def check_if_sample_sizes_are_proportions_or_absolute(sample_groups: Dict[str, Union[float, int]]) -> str:
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
            'The sizes provided as values for `sample_groups` must all be proportions (e.g. 0.8, 0.1, 0.1) or '
            'absolute sizes (e.g. 65, 40, 25).'
        )


def validate_experiment_parameter_between_0_and_1(parameter_value: float, experiment_parameter: str) -> None:
    """
    Raises exception if a parameter for the experiment e.g. power, or significance level is not in the correct range
    0 < parameter_value < 1.

    Parameters
    ----------
    parameter_value :
        Value provided for the parameter e.g. 0.05 to represent a 5% significance level.
    experiment_parameter :
        What type of parameter has been provided e.g. 'significance_level' / 'power' / 'sample_size_proportions'.

    Raises
    -------
    ValueError
        If 0 < parameter_value < 1 not satisfied.
    """
    if not 0 < parameter_value < 1:
        raise ValueError(f"{experiment_parameter} must adhere to 0 < {experiment_parameter} < 1.")


def validate_measurement_type_is_valid(measurement_type: str) -> None:
    """
    Check that the experiment has been correctly specified as measuring proportions or means.

    Parameters
    ----------
    measurement_type : str
        What type of metric the experiment is measuring i.e. proportion (e.g. % conversion rate) or mean (e.g. average
        spend).

    Raises
    ------
    ValueError
        If `measurement_type` not in ['proportion', 'mean'].
    """

    if measurement_type not in ['proportion', 'mean']:
        raise ValueError("The experiment must be measuring a 'proportion' or 'mean'.")


def validate_binary_events_are_represented_with_0_or_1(experiment_observations: np.ndarray) -> None:
    """
    When the experiment is evaluating whether a proportion has changed, the raw observations should be encoded as
    0 (non-event) or 1 (event).

    Parameters
    ----------
    experiment_observations : numpy array_like
        Observations for specific group in the experiment.

    Raises
    ------
    ValueError
        If the observations are not all represented by 0's and 1's only.
    """

    valid_values = {0, 1}
    observed_values = set(experiment_observations)

    if not observed_values.issubset(valid_values):
        raise ValueError(
            'When testing proportions, values must be marked as 1 to represent the event, and 0 for non-event.'
        )


def validate_sample_size_values_are_appropriate(
        original_population: pd.DataFrame,
        sample_groups: Dict[str, Union[float, int]],
        size_type: str,
) -> None:
    """
    When creating experiment groups, check that the values provided for the size of each sample group, be they
    proportions or absolute sizes, are valid.

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
