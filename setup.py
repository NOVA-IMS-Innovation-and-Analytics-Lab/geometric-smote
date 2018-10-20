from setuptools import setup, find_packages

setup(
    name='sklearnext',
    python_requires='>3.5',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'scipy==1.1.0',
        'statsmodels==0.8.0',
        'numpy==1.14.5',
        'pandas==0.23.1',
        'scikit-learn==0.19.1',
        'imbalanced-learn==0.3.3',
        'dask-searchcv==0.2.0',
        'progressbar2==3.38.0'
    ]
)
