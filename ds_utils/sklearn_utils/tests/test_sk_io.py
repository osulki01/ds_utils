"""
Testing for saving and loading sklearn objects.
"""

# Standard library imports
import os
from os import path
import pickle
from pyfakefs.pytest_plugin import fs
import pytest

# Third party imports
import sklearn
from sklearn import preprocessing

# Local application imports
from ds_utils.sklearn_utils import sk_io


@pytest.fixture
def sklearn_current_version():
    """Get current version of sklearn."""
    return sklearn.__version__


@pytest.fixture
def sklearn_scaler_object():
    """Create sklearn object that will be saved and loaded."""
    return preprocessing.StandardScaler()


def test_load_pickled_sklearn_object_and_version(fs, sklearn_scaler_object, sklearn_current_version):
    """
    Both serialised sklearn object and the library version string are successfully loaded.
    """

    # Write object and sklearn version to fake file
    file_path = 'sklearn_object_and_version.pkl'

    sklearn_object_and_version = (sklearn_scaler_object, sklearn_current_version)

    with open(file_path, 'wb') as target_destination:
        pickle.dump(sklearn_object_and_version, target_destination)

    # Load sklearn object and version and check that they are the same values as before
    loaded_sklearn_object, loaded_sklearn_version = sk_io.load_pickled_sklearn_object_and_version(
        filename_or_path=file_path
    )

    assert isinstance(loaded_sklearn_object, type(sklearn_scaler_object))

    assert loaded_sklearn_version == sklearn_current_version




def test_save_pickled_sklearn_object_and_version_handles_existing_directory(fs, sklearn_scaler_object):
    """
    File is created if target directory already exists.
    """

    pre_existing_directory = '/pre_existing_directory'
    os.makedirs(pre_existing_directory)

    sk_io.save_pickled_sklearn_object_and_version(
        sklearn_object=sklearn_scaler_object,
        filename_or_path=f'{pre_existing_directory}/new_sklearn_file',
    )

    assert path.isfile(f'{pre_existing_directory}/new_sklearn_file')


def test_save_pickled_sklearn_object_and_version_handles_non_existing_directory(fs, sklearn_scaler_object):
    """
    File is created if target directory does not already exist.
    """

    non_existing_directory = '/non_existing_directory'

    sk_io.save_pickled_sklearn_object_and_version(
        sklearn_object=sklearn_scaler_object,
        filename_or_path=f'{non_existing_directory}/new_sklearn_file',
    )

    assert path.isfile(f'{non_existing_directory}/new_sklearn_file')


def test_save_pickled_sklearn_object_and_version_handles_existing_file(fs, sklearn_scaler_object):
    """
    Exception is raised if file already exists and overwrite mode is not set, and that file is
    overwritten if it is.
    """

    # Create a fake file
    pre_existing_pathname = '/fake_directory/pre_existing_file'
    fs.create_file(pre_existing_pathname)

    # Check that exception is raised if file already exists and user did not specify it should be over-written
    with pytest.raises(FileExistsError):
        sk_io.save_pickled_sklearn_object_and_version(
            sklearn_object=sklearn_scaler_object,
            filename_or_path=pre_existing_pathname,
            overwrite=False,
        )

    # Check that existing file is overwritten if requested
    existing_file_last_modified_time = os.stat(pre_existing_pathname).st_mtime

    sk_io.save_pickled_sklearn_object_and_version(
        sklearn_object=sklearn_scaler_object,
        filename_or_path=pre_existing_pathname,
        overwrite=True,
    )

    overwritten_file_last_modified_time = os.stat(pre_existing_pathname).st_mtime

    assert overwritten_file_last_modified_time > existing_file_last_modified_time


def test__warn_if_loaded_sklearn_object_version_different_to_current_version():
    """
    User is made aware if the version associated with the loaded sklearn object is different to the version of sklearn
    they are using, or that the version was not originally saved correctly.
    """

    mock_filename_or_path = '/mock_filename.pkl'
    mock_different_sklearn_version = 'mock_different_sklearn_version'

    # Warning should be raised if the version associated with the loaded sklearn object is different to the current
    # sklearn version
    with pytest.warns(UserWarning):
        sk_io._warn_if_loaded_sklearn_object_version_different_to_current_version(
            loaded_sklearn_object_version=mock_different_sklearn_version,
            filename_or_path=mock_filename_or_path,
        )

    # Exception should be raised if the version is not a string, and likely was saved incorrectly
    mock_incorrect_version_type = 999

    with pytest.raises(TypeError):
        sk_io._warn_if_loaded_sklearn_object_version_different_to_current_version(
            loaded_sklearn_object_version=mock_incorrect_version_type,
            filename_or_path=mock_filename_or_path
        )
