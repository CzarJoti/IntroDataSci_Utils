from setuptools import find_packages, setup

setup(
    name='into_datasci_utils',
    install_requires=[
        "torch",
        "scipy",
        "pandas",
        "numpy",
        "scikit-learn"
    ],
    packages=find_packages()
)