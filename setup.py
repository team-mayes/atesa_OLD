import sys
from setuptools import setup, find_packages
import versioneer

with open("README.md", "r") as fh:
    long_description = fh.read()
    
# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

setup(
    name="atesa",
    version="1.0",
    author="Tucker Burgin",
    author_email="tburgin@umich.edu",
    description="Python program for automating Aimless Transition Ensemble Sampling and Analysis (ATESA) with the Amber molecular simulations package.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/team-mayes/atesa",
    packages=find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    scripts=['atesa/atesa.py']
)