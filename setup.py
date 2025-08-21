from setuptools import setup, find_packages

setup(
    name="option_library",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
        "yfinance>=0.1.70"
    ]
) 