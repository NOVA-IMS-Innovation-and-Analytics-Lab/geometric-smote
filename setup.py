from setuptools import setup, find_packages

setup(
    name="imbalanced-tools",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        'docx',
        'numpy',
        'pandas',
        'progressbar2',
        'scipy',
        'scikit_learn',
        'imbalanced-learn'
    ]
)
