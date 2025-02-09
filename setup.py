from setuptools import setup, find_packages

setup(
    name="cluster",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
    ],
    python_requires=">=3.7",
)
