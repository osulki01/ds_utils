import setuptools

package_name = 'ds_utils'
version = '0.1'

setuptools.setup(
    author="Kieran O'Sullivan",
    author_email='osullivank3@hotmail.co.uk',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    description='Utility library for common Data Science tasks which are not handled by existing libraries',
    install_requires=[
        "pandas",
        "scikit-learn",
        "statsmodels",
    ],
    keywords=['data', 'science', 'utilities'],
    license='MIT',
    name=package_name,
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    url='https://github.com/osulki01/ds_utils',
    version=version,
)
