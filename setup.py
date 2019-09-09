import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="liteglm",
    version="0.0.3",
    author="Jose Gama",
    author_email="josephgama@yahoo.com",
    description="GLM methods for robust or incremental GLM, based on Mike Kane and Bryan W. Lewis code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tuxcell/liteglm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
