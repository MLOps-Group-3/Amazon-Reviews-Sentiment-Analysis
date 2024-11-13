from setuptools import setup, find_packages

setup(
    name='sentiment_analysis_utils',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'torch',
        'transformers',
        'mlflow',
    ],
)
