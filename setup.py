from setuptools import setup, find_packages

setup(
    name="automac",
    version="0.1.1",
    author="Abhishek Sharma",
    description="An all-in-one automated ML pipeline for feature engineering, optimization, and evaluation.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jubito-27/ml-automator",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "category_encoders",
        "optuna",
        "catboost",
        "xgboost",
        "lightgbm",
        "plotly",
        "kaleido",
        "openpyxl"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)

