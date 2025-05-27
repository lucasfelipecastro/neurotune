from setuptools import setup, find_packages

setup(
    name="neurotune",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.32.0",
        "transformers>=4.37.2",
        "torch>=2.2.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
    ],
) 