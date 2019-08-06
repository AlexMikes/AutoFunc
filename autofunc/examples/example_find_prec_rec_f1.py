from autofunc.get_match_factor import match
from autofunc.get_precision_recall import precision_recall
from autofunc.simple_counter import count_stuff
from autofunc.get_top_results import get_top_results
from autofunc.get_data import get_data
from autofunc.counter_pandas import counter_pandas
from autofunc.make_df import make_df
from autofunc.find_associations import find_associations
import os.path

""" Example showing how to find the match factor using the simple counting file """


# Dataset used for data mining
script_dir = os.path.dirname(__file__)
file1 = os.path.join(script_dir, '../assets/bladeCombined_id.csv')

# Use a known product for verification
test_file = os.path.join(script_dir, '../assets/jigsawQuery_headers.csv')

test_data, test_records, test_data_no_ids, test_records_no_ids = get_data(test_file)


# Use a threshold to get the top XX% of confidence values
threshold = 0.69


# Counting:
# comb_sort = count_stuff(file1)
# thresh_results = get_top_results(comb_sort, threshold)


# Association Rules:
# store_data, records, store_data_no_ids, records_no_ids = get_data(file1)
# conf_results, results = find_associations(store_data_no_ids, records_no_ids)
# thresh_results = get_top_results(conf_results, threshold)


# Pandas
# df = make_df(file1)
# comb_sort = counter_pandas(df)
# thresh_results = get_top_results(comb_sort, threshold)


# Find the match factor of the verification test by comparing the learned results with the known function/flows
learned_dict, matched, overmatched, unmatched, recall, precision, f1 = precision_recall(thresh_results, test_records)

print('Recall = {0:.5f}'.format(recall))
print('Precision = {0:.5f}'.format(precision))
print('F1 = {0:.5f}'.format(f1))


