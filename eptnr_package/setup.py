import setuptools
import os

# Obtained from https://stackoverflow.com/a/53069528
lib_folder = os.path.dirname(os.path.abspath(__file__)) or '.'
requirement_path = lib_folder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

setuptools.setup(
    name="eptnr",
    version="0.0.1",
    author="Riccardo Fiorista",
    author_email="rfiorista@uva.nl",
    description="A package for Public Transport Network Reduction Under Equality",
    long_description_content_type="text/markdown",
    url="https://github.com/NONE_YET",
    project_urls={
        "Bug Tracker": "https://github.com/NONE_YET/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.8",
    install_requires=install_requires,
)
