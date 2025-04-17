"""
Setup script for the Heart Failure Detection project.
"""

from setuptools import setup, find_packages

setup(
    name="heart_failure_detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy==1.26.4",
        "pandas==2.2.2",
        "matplotlib==3.9.0",
        "seaborn",
        "scikit-learn",
        "imbalanced-learn",
        "lightgbm",
        "xgboost",
        "catboost==1.2.6",
        "shap",
        "scikit-optimize",
        "optuna",
        "joblib",
        "flask",
        "pytest",
        "pyreadstat",
        "gdown"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A machine learning project for early detection of heart failure",
    keywords="machine learning, heart failure, healthcare",
    url="https://github.com/yourusername/heart-failure-detection",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
