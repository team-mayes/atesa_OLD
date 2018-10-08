import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="aimless_shooting",
    version="1.0",
    author="Tucker Burgin",
    author_email="tburgin@umich.edu",
    description="Python code for performing aimless shooting for transition path sampling in Amber.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/team-mayes/aimless_shooting",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    scripts=['aimless_shooting/aimless_shooting.py'],
    install_requirements=['Jinja2==2.9.5', 'pytraj==2.0.2']
)