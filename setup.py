from setuptools import setup, find_packages

setup(
    name='sklearnext',
    python_requires='>3.5',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'scipy==1.1.0',
        'numpy==1.15.3',
        'pandas==0.23.4',
        'scikit-learn==0.20.0',
        'imbalanced-learn==0.4.2',
        'statsmodels==0.9.0',
        'somoclu==1.7.5',
        'dask-searchcv==0.2.0',
        'progressbar2==3.38.0'
    ]
)
