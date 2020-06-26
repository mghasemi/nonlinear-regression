try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

Description = "This repository contains a structured code that facilitates non-linear regression based on scikit-learn."

setup(
    name="NonlinearRegression",
    version="0.0.1",
    author="Mehdi Ghasemi",
    author_email="mehdi.ghasemi@gmail.com",
    packages=["NonlinearRegression"],
    url="https://gitlab.com/mghasemi/nonlinear-regression",
    license="MIT License",
    description=Description,
    #long_description=open("./readme.rst").read(),
    keywords=[
        "Optimization",
        "Numerical",
        "Machine Learning",
        "Regression",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-learn",
    ],
)
