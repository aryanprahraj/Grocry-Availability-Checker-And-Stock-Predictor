from setuptools import setup, find_packages

setup(
    name="grocery-stock-predictor-ml",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
    ],
    author="Aditya Bhuran",
    description="Grocery availability checker and stock predictor",
    python_requires=">=3.8",
)
