from autofunc.get_match_factor import match
from autofunc.get_top_results import get_top_results
from autofunc.find_associations import find_associations
from autofunc.get_data import get_data
import os.path

""" Example showing how to find the match factor using association rules """


# Dataset used for data mining
script_dir = os.path.dirname(__file__)
file1 = os.path.join(script_dir, '../assets/bladeCombined_id.csv')

# Convert file to data frame and list
store_data, records = get_data(file1)

# Use Association Rules to sort the functions/flows of components by confidence
conf_results, results = find_associations(store_data, records)

# Use a threshold to get the top XX% of confidence values
thresh_results = get_top_results(conf_results, 0.69)

# Use a known product for verification
test_file = os.path.join(script_dir, '../assets/jigsawQuery_headers.csv')

test_data, test_records = get_data(test_file)

# Find the match factor of the verification test by comparing the learned results with the known function/flows
learned_dict, matched, overmatched, unmatched, match_factor = match(thresh_results, test_records)

print('Match factor = {0:.5f}'.format(match_factor))