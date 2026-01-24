#setup.py
from setuptools import find_packages, setup

setup(
    name="home-price-prediction",
    version="0.1.0",
    description="Build ML project that predicts property prices in rupees",
    author="bhushan",
    packages=find_packages(),
    install_requires=[], # You can leave this empty since you use requirements.txt
)