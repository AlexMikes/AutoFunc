from autofunc.get_match_factor import match
from autofunc.get_precision_recall import precision_recall
from autofunc.simple_counter import count_stuff
from autofunc.get_top_results import get_top_results
from autofunc.get_data import get_data
from autofunc.counter_pandas import counter_pandas
from autofunc.make_df import make_df
from autofunc.split_learning_verification import split_learning_verification
from autofunc.df_to_list import df_to_list
from autofunc.find_associations import find_associations
import os.path

""" Example showing how to find the match factor using the simple counting file """

#
# # Dataset used for data mining
# script_dir = os.path.dirname(__file__)
# file1 = os.path.join(script_dir, '../assets/bladeCombined_id.csv')
#
# # Use a known product for verification
# test_file = os.path.join(script_dir, '../assets/jigsawQuery_headers.csv')
#
# test_data, test_records, test_data_no_ids, test_records_no_ids = get_data(test_file)

script_dir = os.path.dirname(__file__)
file1 = os.path.join(script_dir, '../assets/new_blade.csv')


# Use a threshold to get the top XX% of confidence values
threshold = 0.5

# Pandas
df = make_df(file1)
comb_sort = counter_pandas(df)
thresh_results = get_top_results(comb_sort, threshold)


verification_ids  = [376, 608]


ver_df, learn_df = split_learning_verification(df, verification_ids)

ver_list = df_to_list(ver_df)


# Find the match factor of the verification test by comparing the learned results with the known function/flows
learned_dict, matched, overmatched, unmatched, recall, precision, f1 = precision_recall(thresh_results, ver_list)

print('Recall = {0:.5f}'.format(recall))
print('Precision = {0:.5f}'.format(precision))
print('F1 = {0:.5f}'.format(f1))


