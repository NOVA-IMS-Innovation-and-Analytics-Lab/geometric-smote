from setuptools import setup, find_packages

setup(
    name='gsmote',
    python_requires='>3.5',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'scipy>=0.17',
        'numpy>=1.1',
        'scikit-learn>=0.21',
        'imbalanced-learn>=0.4.3'
    ]
)
