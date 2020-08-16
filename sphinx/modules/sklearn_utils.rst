.. sklearn_utils:

ds_utils.sklearn_utils
======================

Helper functions when working with the Scikit-learn library.

.. note::
   The terms Scikit-learn and sklearn are used interchangeably.


Contents
~~~~~~~~

.. contents::
   :local:
   :depth: 1


Saving and Loading Scikit-learn Objects
---------------------------------------

**Saving**

When working with Scikit-learn, and machine learning in general, different versions of the library you are using can
yield different outputs or inconsistent functionality.

To solve the problem of "it works on my machine", it is generally recommended to make use of
`Docker <https://www.docker.com/why-docker>`_, or at least a virtual environment.

However, there may still be occasions when working with Scikit-learn models that you have inherited from a colleague,
or even one that you have worked on previously, when you are left wondering what version of Scikit-learn it was
originally created with.

:py:func:`ds_utils.sk_io.save_pickled_sklearn_object_and_version` can be used to save both the
Scikit-learn object, and the version of Scikit-learn currently running.

.. code-block:: python

   >>> from sklearn import preprocessing
   >>> from ds_utils.sklearn_utils import sk_io

   >>> data = [[0, 0], [0, 0], [1, 1], [1, 1]]
   >>> scaler = preprocessing.StandardScaler()
   >>> scaler.fit(data)

   >>> sk_io.save_pickled_sklearn_object_and_version(scaler, 'scaler_and_sklearn_version.pkl')


In order to overwrite an existing file, you must request :py:data:`overwrite` mode.

.. code-block:: python
   :emphasize-lines: 9

   # Attempt to overwrite an existing file
   >>> sk_io.save_pickled_sklearn_object_and_version(scaler, 'scaler_and_sklearn_version.pkl')
   FileExistsError: File scaler_and_sklearn_version.pkl already exists.

   # Successfully overwrite an existing file
   >>> sk_io.save_pickled_sklearn_object_and_version(
   >>>   sklearn_object=scaler,
   >>>   filename_or_path='scaler_and_sklearn_version.pkl',
   >>>   overwrite=True,
   >>> )


.. warning::
   Only use :py:func:`ds_utils.sk_io.save_pickled_sklearn_object_and_version` when saving an sklearn object you have
   trained/created yourself, not an already-pickled object that you have loaded from elsewhere, as you do not know
   whether your version of sklearn is the same as the one used to originally create it.


**Loading**

Your Scikit-learn object and the running version of Scikit-learn have now been saved together as a Tuple in a
`pickled <https://docs.python.org/3/library/pickle.html>`_ object.

They can now be loaded using :py:func:`ds_utils.sk_io.save_pickled_sklearn_object_and_version`. This method will issue
a warning if the version associated with the loaded object differs to the one that you are currently warning.

.. code-block:: python

   >>> loaded_sklearn_object, loaded_sklearn_version = sk_io.load_pickled_sklearn_object_and_version(
   >>>   filename_or_path='scaler_and_sklearn_version.pkl'
   >>> )

   >>> print(type(loaded_sklearn_object))
   <class 'sklearn.preprocessing._data.StandardScaler'>

   >>> print(loaded_sklearn_version)
   0.23.1


Module Overview
---------------

.. autosummary::

   ds_utils.sklearn_utils.sk_io


Submodules
----------

sk_io
^^^^^

.. automodule:: ds_utils.sklearn_utils.sk_io
   :members:
