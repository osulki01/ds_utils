"""
Testing for checking that the parameters provided when setting up or evaluating an experiment are valid.
"""

# Third party imports
import numpy as np
import pandas as pd
import pytest

# Local application imports
from ds_utils.hypothesis_testing import _check_experiment_inputs


@pytest.mark.parametrize(
    'inconsistent_sample_size_types',
    [
        {'Group_1': 0.5, 'Group_2': 25, 'Group_3': 25},
        {'Group_1': 50, 'Group_2': 0.25, 'Group_3': 25},
        {'Group_1': 50, 'Group_2': 25, 'Group_3': 0.25},
    ]
)
def test_check_if_sample_sizes_are_proportions_or_absolute_flags_inconsistent_size_type(
        inconsistent_sample_size_types
):
    """
    When the size of each sample group, all of the size must be floats (for proportions) or integers (for absolute
    sizes).
    """

    with pytest.raises(
            ValueError,
            match='The sizes provided as values for `sample_groups` must all be proportions .* or absolute sizes .*'
    ):
        _check_experiment_inputs.check_if_sample_sizes_are_proportions_or_absolute(inconsistent_sample_size_types)


def test_check_if_sample_sizes_are_proportions_or_absolute_recognises_absolute_sizes():
    """Recognise that integers represent absolute number of records to be assigned to each sample group."""

    sample_groups = {'Group_1': 50, 'Group_2': 25, 'Group_3': 25}
    assert _check_experiment_inputs.check_if_sample_sizes_are_proportions_or_absolute(sample_groups) == 'absolute'


def test_check_if_sample_sizes_are_proportions_or_absolute_recognises_proportions():
    """Recognise that floats represent proportions of the population to be assigned to each sample group."""

    sample_groups = {'Group_1': 0.5, 'Group_2': 0.25, 'Group_3': 0.25}
    assert _check_experiment_inputs.check_if_sample_sizes_are_proportions_or_absolute(sample_groups) == 'proportion'


@pytest.mark.parametrize(
    'invalid_sample_sizes, size_type',
    [
        ({'Group_1': 0.5, 'Group_2': 0.5, 'Group_3': 0.5}, 'proportion'),  # Sum of proportions greater than 1
        ({'Group_1': -0.5, 'Group_2': 0.25, 'Group_3': 0.25}, 'proportion'),  # One proportion value <= 0
        ({'Group_1': 1.5, 'Group_2': 0.25, 'Group_3': 0.25}, 'proportion'),  # One proportion value >= 1
        ({'Group_1': 3, 'Group_2': 3, 'Group_3': 40}, 'absolute'),  # Overall size exceeds that of the population
    ]
)
def test_validate_sample_size_values_are_appropriate(invalid_sample_sizes, size_type):
    """
    Sample size proportions should not exceed 1 in total, and should all be 0 < proportion < 1.
    Absolute sizes should not exceed the total population when summed together.
    """

    original_population_10_rows = pd.DataFrame(np.arange(10), columns=['original_row_index'])

    expected_exception_messages = r'(Proportions should all be 0 < proportion < 1.)|' \
                                  r'(Proportions should not exceed 1 in total.)|' \
                                  r'(The sum of all sample sizes should not exceed that of the original population .*)'

    with pytest.raises(ValueError, match=expected_exception_messages):
        _check_experiment_inputs.validate_sample_size_values_are_appropriate(
            original_population=original_population_10_rows,
            sample_groups=invalid_sample_sizes,
            size_type=size_type,
        )


@pytest.mark.parametrize('invalid_parameter_value', [-1, 0, 1])
def test_validate_experiment_parameter_between_0_and_1(invalid_parameter_value):
    """
    Certain experiment paramaters, such as power or level of significance are a percentage, and an exception should be
    raised if condition 0 < parameter < 1 not met.
    """
    with pytest.raises(ValueError, match=".* must adhere to 0 < .* < 1."):
        _check_experiment_inputs.validate_experiment_parameter_between_0_and_1(
            parameter_value=invalid_parameter_value,
            experiment_parameter='experiment_parameter'
        )


def test_validate_measurement_type_is_valid():
    """The experiment's metric should be measuring proportions or means."""

    with pytest.raises(ValueError, match="The experiment must be measuring a 'proportion' or 'mean'."):
        _check_experiment_inputs.validate_measurement_type_is_valid('invalid_measurement_type')


def test_validate_binary_events_are_represented_with_0_or_1():
    """When testing proportions, values must be marked as 1 to represent the event, and 0 for non-event."""

    invalid_proportions = np.array([0, 1, 0, 1, 2])

    with pytest.raises(
            ValueError,
            match='When testing proportions, values must be marked as 1 to represent the event, and 0 for non-event.'
    ):
        _check_experiment_inputs.validate_binary_events_are_represented_with_0_or_1(invalid_proportions)
