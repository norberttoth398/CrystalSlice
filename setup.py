
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CrystalSlice",
    version="0.0.1",
    author="norberttoth398",
    author_email="nt398@cam.ac.uk",
    description="Slicing numerical model.",
    long_description=long_description,
    url="https://github.com/norberttoth398/CrystalSlice",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
