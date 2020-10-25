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

## Dependencies

This package uses Pandas (Python Data Analysis Library). It can be installed with pip using:

```bash
pip install pandas
```
Many of the examples also use Matplotlib for plotting. While not required to use the AutoFunc modules, it is required to run the examples. It can be installed with:

```bash
pip install -U matplotlib
```

## Usage

Example files are provided in the examples folder. Autofunc will automate the functional representations of components
as  long as the format of the .csv file is has the component in column 1 and the function-flow in column 2

More information on the methods used in these files can found in the various research papers that this software supports, especcially IDETC2020-22346
"OPTIMIZING AN ALGORITHM FOR DATA MINING A DESIGN REPOSITORY TO AUTOMATE FUNCTIONAL MODELING". All of the plots for this paper were created in the ```example_optimize_with_comp_ratio.py``` file.

The following lists the examples included, with their expected functionality and outputs:

1. ```example_cross_validation.py``` uses the k-fold cross validation functionality to find the accuracy of a data mining classifer. This example will print the maximum and average accuracies using this verification method.

1. ```example_find_f1_from_file.py``` finds the F1 score of a single product when the component-function-flow combinations for that product are in a separate .csv file. This example will print the Recall, Precision, and F1 score for that testing product.

1. ```example_find_f1_from_id.py``` finds the F1 score of a single product using that product's ID number from the original dataset. Any number of IDs can be used. This example will print the testing ID(s) used, and the recall, precision, and F1 score for those testing IDs.

1. ```example_find_similarity.py``` will create a similarity matrix for the training dataset. This is the percent of similar components between each product in the dataset. The main diagonal of this matrix consists of ones because every product is 100% similar to itself, but the matrix is not symmetric because each product can contain a different number of components. For example, consider a case where Product 1 has 20 components and Product 2 has 40 components. If they have 10 components in common, the similarity between Product 1 and Product 2 is 10/20 = 50%, but the similarity between Product 2 and Product 1 is 10/40 = 25%. The first product of the pair is known as the ”generating” product, which is the product in the column of this matrix. This example will create a Pandas dataframe of the similarity matrix and write this to a .csv file.

1. ```example_get_func_rep.py``` will create a functional representation of the components in the input file using data mining and a classification threshold. This can be used to automate functional modeling by connecting the functions and flows at the interface of components in a product. This example will write a .csv file with the results of component-function-flow and optional frequency.

1. ```example_optimization.py``` incorporates all of the main modules and optimizes the similarity and classification thresholds. This example will display a lot of plots and print optimum values for thresholds.

1. ```example_optimize_with_comp_ratio.py``` begins with ```example_optimization.py``` and also includes the stratification and optimization of a training set. This example will display a lot of plots and print optimum values for thresholds.

1. ```example_try_best_ids.py``` is a subset of ```example_optimize_with_comp_ratio.py``` which only includes the stratified training set and some
relevant plots. This example will display a plot of the F1 scores vs. Classification threshold of the stratified dataset.


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
write_results_from_dict(results, 'test1.csv')
```


Run from within ```examples``` folder:

```bash
python example_get_func_rep.py
```

And it will generate a file ```test1.csv``` with the results of the automated functional representation of the 
 components in the ```input_file``` based on the data from the ```file_to_learn``` in the ```assets``` folder.
 
## Testing
All tests are automated through [Travis CI](https://travis-ci.org/). Visit [this page](https://travis-ci.org/github/AlexMikes/AutoFunc) to view the results.


## Support
Please submit requests for support or problems with software as issues in the repository.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
