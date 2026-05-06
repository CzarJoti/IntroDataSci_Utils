from setuptools import find_packages, setup

setup(
    name='intro_datasci_utils',
    install_requires=[
        "torch",
        "scipy",
        "pandas",
        "numpy",
        "scikit-learn"
    ],
    packages=find_packages()
)