from setuptools import setup, find_packages

setup(
    name="sklearnext",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'scipy',
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'imbalanced-learn',
        'dask_ml',
        'progressbar2'
    ]
)
