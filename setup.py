# setup.py
from setuptools import setup, find_packages

setup(
    name="kadar",
    version="1.0.0",
    author="Madeleine Lutze",
    description="A Python package for predicting HGT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "sourmash>=4.8.0"
    ],
)