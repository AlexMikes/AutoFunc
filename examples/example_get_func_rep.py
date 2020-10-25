from autofunc.get_top_results import get_top_results
from autofunc.counter_pandas import counter_pandas
from autofunc.get_func_rep import get_func_rep
from autofunc.write_results import write_results_from_dict
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
write_results_from_dict(combos_sorted, 'func_rep1.csv')