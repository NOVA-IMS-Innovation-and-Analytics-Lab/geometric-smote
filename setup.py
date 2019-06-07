from setuptools import setup, find_packages

setup(
    name='sklearnext',
    python_requires='>3.5',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'scipy==1.2.1',
        'numpy==1.16.3',
        'pandas==0.24.2',
        'scikit-learn==0.20.3',
        'imbalanced-learn==0.4.3',
        'statsmodels==0.9.0',
        'somoclu==1.7.5'
    ]
)
