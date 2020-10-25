from autofunc.get_top_results import get_top_results
from autofunc.counter_pandas import counter_pandas
from autofunc.get_precision_recall import precision_recall
from autofunc.df_to_list import df_to_list
import os.path
import pandas as pd

""" Example showing how to find F1 score using separate file of input components """


# Dataset used for data mining
script_dir = os.path.dirname(__file__)
file_to_learn = os.path.join(script_dir, '../autofunc/assets/consumer_systems.csv')

train_data = pd.read_csv(file_to_learn)
combos_sorted = counter_pandas(train_data)

# Use a threshold to get the top XX% of confidence values
threshold = 0.5
thresh_results = get_top_results(combos_sorted, threshold)

# Use a known product for verification
test_file = os.path.join(script_dir, '../autofunc/assets/jigsawQuery_headers.csv')
test_data = pd.read_csv(test_file)
test_list = df_to_list(test_data)

learned_dict, matched, overmatched, unmatched, recall, precision, f1 = precision_recall(thresh_results,
                                                                                                    test_list)


# Optional write to file - uncomment and rename to write file
# write_results_from_dict(learned_dict, 'test1.csv')



print('Recall = {0:.5f}'.format(recall))
print('Precision = {0:.5f}'.format(precision))
print('F1 = {0:.5f}'.format(f1))


