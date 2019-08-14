import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="atesa",
    version="1.0",
    author="Tucker Burgin",
    author_email="tburgin@umich.edu",
    description="Automates performing and analyzing the results of aimless shooting transition path sampling in Amber.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/team-mayes/atesa",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    scripts=['atesa/atesa.py'],
    install_requirements=['jinja2>=2.9.5', 'pytraj>=2.0.2', 'statsmodels', 'pathos']
)