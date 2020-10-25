"""

This is a subset of "example_optimize_with_comp_ratio.py" which only includes the stratified training set and some
relevant plots

"""

from autofunc.get_precision_recall import precision_recall
from autofunc.get_top_results import get_top_results
from autofunc.make_df import make_df
from autofunc.find_similarities import find_similarities
from autofunc.counter_pandas import counter_pandas
from autofunc.split_learning_verification import split_learning_verification
from autofunc.df_to_list import df_to_list
import os.path
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
import matplotlib.pyplot as plt
from operator import itemgetter
import numpy as np
from mpl_toolkits import mplot3d
import time
import pandas as pd
from statistics import mean


start = time.time()

# Dataset used for data mining
script_dir = os.path.dirname(__file__)
file_to_learn = os.path.join(script_dir, '../autofunc/assets/consumer_systems.csv')

train_data = pd.read_csv(file_to_learn)
train_df_whole = make_df(file_to_learn)

all_train_ids = list(map(int,train_data.id.unique()))

f1s = []
save_data = []

best_ids = [319, 370, 380, 603, 364, 225, 652, 689, 688, 697, 609, 357, 712, 605, 208, 606, 206, 345, 335, 599, 601, 615, 686, 358, 376, 366, 670, 334, 305, 671]

# Remove best_ids from all_train_ids
test_ids = [x for x in all_train_ids if x not in best_ids]

train_df = train_df_whole[train_df_whole['id'].isin(best_ids)]

comb_sort = counter_pandas(train_df)

for test_id in test_ids:

    test_df = train_df_whole[train_df_whole['id']==test_id]
    test_list = df_to_list(test_df)

    for t in range(10, 100, 5):
        threshold = t / 100
        print(test_id, ' ', threshold)

        thresh_results = get_top_results(comb_sort, threshold)

        # Find the F1 score of the verification test by comparing the learned results with the known function/flows
        learned_dict, matched, overmatched, unmatched, recall, precision, f1 = precision_recall(thresh_results,
                                                                                                test_list)
        # num_train_comps = len(train_comps)

        save_data.append((test_id, threshold, f1))

        # points.append((ps_thresh, threshold, f1))

        f1s.append(f1)

all_data = pd.DataFrame(save_data,columns = ['Test Product ID', 'Thresh','F1'])

thresh_plot = []
avg_f1s = []
for t in range(10, 100, 5):
    threshold = t/100
    avg_f1s.append(mean(all_data['F1'][(all_data['Thresh'] == threshold)]))
    thresh_plot.append(threshold)

#Plotting f1 vs num ids
plt.plot(thresh_plot,avg_f1s)
plt.ylabel('Average F1 Score')
plt.xlabel('Classification Threshold')
plt.ylim(0.19,0.46)
# plt.title('Avg F1 score vs Number of Products')
plt.grid()
plt.show()

############################



