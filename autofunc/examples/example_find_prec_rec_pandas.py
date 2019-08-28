from autofunc.get_match_factor import match
from autofunc.get_precision_recall import precision_recall
from autofunc.simple_counter import count_stuff
from autofunc.get_top_results import get_top_results
from autofunc.get_data import get_data
from autofunc.counter_pandas import counter_pandas
from autofunc.make_df import make_df
from autofunc.split_learning_verification import split_learning_verification
from autofunc.df_to_list import df_to_list
from autofunc.write_results import write_results_from_dict
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
file1 = os.path.join(script_dir, '../assets/consumer_systems.csv')


# Use a threshold to get the top XX% of confidence values
threshold = 0.7

# Pandas
df = make_df(file1)



### Not B&D
verification_ids  = [691]
#
ver_df, learn_df = split_learning_verification(df, verification_ids)
#
# # ver_df.to_csv('test_df_consumer.csv')
#
ver_list = df_to_list(ver_df)


# Learning

comb_sort = counter_pandas(learn_df)
thresh_results = get_top_results(comb_sort, threshold)

# Find the match factor of the verification test by comparing the learned results with the known function/flows
learned_dict, matched, overmatched, unmatched, recall, precision, f1 = precision_recall(thresh_results, ver_list)


# # ### B&D
# #
# file2 = os.path.join(script_dir, '../assets/bd_systems.csv')
#
# # Learning
# bd_df = make_df(file2)
# bd_comb_sort = counter_pandas(bd_df)
# bd_thresh_results = get_top_results(bd_comb_sort, threshold)
# #
# ver_list = df_to_list(ver_df)
#
# learned_dict, matched, overmatched, unmatched, recall, precision, f1 = precision_recall(bd_thresh_results, ver_list)
# #

write_results_from_dict(thresh_results, 'consumer_results_70.csv')



print('Recall = {0:.5f}'.format(recall))
print('Precision = {0:.5f}'.format(precision))
print('F1 = {0:.5f}'.format(f1))


