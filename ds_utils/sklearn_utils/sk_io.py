import os
from os import path
import pickle
import sklearn
from typing import Tuple
import warnings


def load_pickled_sklearn_object_and_version(filename_or_path: str) -> Tuple:
    """
    Load a pickled sklearn object and the sklearn version associated with when the sklearn object was saved.

    Parameters
    ----------
    filename_or_path : str
        Location where the sklearn object and its version is saved.

    Returns
    -------
    Tuple[sklearn_object, str]
        sklearn_object: sklearn object.
        str: Version of sklearn when the object was saved.

    Raises
    ----------
    UserWarning
        If the sklearn version associated with the loaded object is different to the current version of sklearn being
        used.
    """

    with open(filename_or_path, 'rb') as file_to_load:
        sklearn_object, sklearn_version = pickle.load(file_to_load)

    _warn_if_loaded_sklearn_object_version_different_to_current_version(
        loaded_sklearn_object_version=sklearn_version,
        filename_or_path=filename_or_path,
    )

    return sklearn_object, sklearn_version


def save_pickled_sklearn_object_and_version(
        sklearn_object,
        filename_or_path: str,
        overwrite: bool = False
) -> None:
    """
    Saves sklearn object as a pickle file, along with the running version of the sklearn library.

    Should only be used to save an sklearn object you have trained/created yourself, not an already-pickled object that
    you have loaded, as it will save the sklearn version that you are currently running rather than the one used to
    originally create the sklearn object.

    Parameters
    ----------
    sklearn_object : sklearn object
        Model/sklearn-object to be saved.
    filename_or_path : str
        Target where the object and its version will be saved.
    overwrite : bool (default is False)
        Whether to overwrite file if it already exists.

    Raises
    ----------
    FileExistsError
        If a `filename_or_path` already exists and user did not set `overwrite` mode.
    """

    # Exit if file already exists and user did not choose to overwrite
    if path.exists(filename_or_path) and not overwrite:
        raise FileExistsError(f'File {filename_or_path} already exists. \nTo overwrite an existing file, '
                              f'set overwrite=True when calling this method.')

    else:
        sklearn_object_and_version = (sklearn_object, sklearn.__version__)

        # If the directory does not already exist, then create it
        os.makedirs(path.dirname(filename_or_path), exist_ok=True)

        # Save both the sklearn object and version of sklearn
        with open(filename_or_path, 'wb') as target_destination:
            pickle.dump(sklearn_object_and_version, target_destination)


def _warn_if_loaded_sklearn_object_version_different_to_current_version(
        loaded_sklearn_object_version: str,
        filename_or_path: str
) -> None:
    """
    Throw a warning if the version associated with an sklearn object that has been loaded is different to the current
    version running.

    Parameters
    ----------
    loaded_sklearn_object_version : str
        Version of sklearn associated with the loaded object.
    filename_or_path : str
        Location where the sklearn object and its version is saved.
    """

    sklearn_current_version = sklearn.__version__

    # Raise exception if the sklearn version was not saved correctly and is not a string
    if not isinstance(loaded_sklearn_object_version, str):
        raise TypeError(f"Version of sklearn associated with loaded object is not a string. \nCheck that the pickled "
                        f"file being loaded was saved in the correct format and order: Tuple[sklearn_object, "
                        f"str]. \nFile to be checked: {filename_or_path}")

    # Warn user if the version associated with the loaded sklearn object is different to the current sklearn version
    if loaded_sklearn_object_version != sklearn_current_version:
        warnings.warn(
            message=f"""
The version of sklearn used when saving the original sklearn object is different to your current version.
Version associated with the loaded sklearn object: {loaded_sklearn_object_version}
Current version: {sklearn_current_version}
""",
            category=UserWarning
        )
