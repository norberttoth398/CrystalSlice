# CrystalSlice
[![codecov](https://codecov.io/gh/norberttoth398/CrystalSlice/graph/badge.svg?token=ZHEVENILZP)](https://codecov.io/gh/norberttoth398/CrystalSlice)

Repo for numerical model of cuboid and crystal slicing.

## Install

To install, you must be a user who can access this private repo (you are if you are one reading this!) and you must have an environment set up using python 3.8 - either a conda or venv environment will suffice.

#### Step 1
Activate environment and install PyMatGen: 

	eg. conda activate CrystalSliceEnv
	pip install pymatgen

Note, this has to be done separate as PyMatGen has been found to be temperamental depending on OS. Works fine as part of requirements in windows but not Linux at times. Can also use conda.

#### Step 2
Install package: 

	pip install git+https://git@github.com/norberttoth398/CrystalSlice#egg=CrystalSlice
