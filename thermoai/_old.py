from setuptools import setup, find_packages

setup(
    name="thermoai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxlib",
        "numpy",
        "scipy",
        "equinox",
        "optax",
        "diffrax",
        "pytest"
    ],
    author="Geoffrey Roeder",
    author_email="roeder@princeton.edu",
    description="A package for research into computational design of thermodynamic AI",
)