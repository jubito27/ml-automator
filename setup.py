from setuptools import setup, find_packages

setup(
    name="automac",
    version="0.1.2", # Version bump because of major feature additions
    author="Abhishek Sharma",
    description="An all-in-one automated ML pipeline for advanced feature engineering, Boruta selection, and parallel optimization.",
    long_description=open("README.md" , encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jubito-27/ml-automator",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn>=1.0",
        "category_encoders",
        "optuna",
        "catboost",
        "xgboost",
        "lightgbm",
        "boruta",
        "plotly",
        "kaleido",
        "openpyxl",
        "matplotlib",
        "seaborn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
)