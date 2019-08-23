import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

setuptools.setup(
    name="atesa",
    version="1.0",
    author="Tucker Burgin",
    author_email="tburgin@umich.edu",
    description="Automates performing and analyzing the results of aimless shooting transition path sampling with Amber.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/team-mayes/atesa",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    scripts=['atesa/atesa.py']
)