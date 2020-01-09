# AutoFunc
Data Mining for Automated Functional Representations

[![Build Status](https://travis-ci.org/AlexMikes/AutoFunc.svg?branch=master)](https://travis-ci.org/AlexMikes/AutoFunc)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3243689.svg)](https://doi.org/10.5281/zenodo.3243689)

This package automatically generates functional representations for components based on the results of data mining a
design repository. It was developed for use with the Design Repository house at Oregon State University. A rudimentary 
web interface can be found here: http://ftest.mime.oregonstate.edu/repo/browse/

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install autofunc.

The package is not yet on PyPI, so it must be downloaded from here as a .zip file: https://github.com/AlexMikes/AutoFunc

Once downloaded as a .zip file, install with:

```bash
pip install /path/to/file/AutoFunc-master.zip
```

## Usage

Example files are provided in the examples folder. Autofunc will automate the functional representations of components
as  long as the format of the .csv file is has the component in column 1 and the function-flow in column 2


This is the ```example_get_func_rep.py``` file:

```python
from autofunc.get_top_results import get_top_results
from autofunc.counter_pandas import counter_pandas
from autofunc.get_func_rep import get_func_rep
import os.path
import pandas as pd


""" Example showing how to automate functional representation """

# Dataset used for data mining
script_dir = os.path.dirname(__file__)
file_to_learn = os.path.join(script_dir, '../assets/consumer_systems.csv')

include_frequencies = True

train_data = pd.read_csv(file_to_learn)
combos_sorted = counter_pandas(train_data)

# Use a threshold to get the top XX% of confidence values
threshold = 0.5
thresh_results = get_top_results(combos_sorted, threshold)

# Use a known product for verification
input_file = os.path.join(script_dir, '../assets/InputExample.csv')

# Get dictionary of functions and flows for each component based on data mining
results, unmatched = get_func_rep(thresh_results, input_file, include_frequencies)


# Optional write to file - uncomment and rename to write file
# write_results_from_dict(results, 'test1.csv')
```


Run from within examples folder:

```bash
python example_get_func_rep.py
```

And it will generate a file ```test1.csv``` with the results of the automated functional representation of the 
 components in the ```input_file``` based on the data from the ```file_to_learn``` in the ```assets``` folder.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)