from setuptools import setup, find_packages

setup(
    name="metric-learn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'scikit-learn',
        'imbalanced-learn'
    ]
)
