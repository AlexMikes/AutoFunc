"""

Example using k-fold cross-validation to find the accuracy of a data mining classifier

The products in the folds are randomized, so it calculates an average of multiple iterations of the cross-validation
To change the number of folds, change the variable "k = 10" to a different number
To change the number of iterations folds that are averaged, change the variable "iters = 10" at the top of the main loop

This can use separate sets of data for training and testing sets. The training set comes from the first file called
"file_to_learn" and the testing set comes from the second file called "file_to_test"

If using the Black and Decker as the training set, set the boolean near the top to "bd = True" and it will use a
different loop that changes all of the necessary variables

"""



from autofunc.get_precision_recall import precision_recall
from autofunc.get_top_results import get_top_results
from autofunc.counter_pandas import counter_pandas
from autofunc.counter_pandas_with_counts import counter_pandas_with_counts
from autofunc.make_df import make_df
from autofunc.split_learning_verification import split_learning_verification
from autofunc.df_to_list import df_to_list
import os.path
from math import floor
import random
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from operator import itemgetter
from statistics import mean
import numpy as np
from mpl_toolkits import mplot3d
import scipy.stats as stats

import time


start = time.time()

## If using the black and decker dataset, set this to True
bd = False

# Dataset used for data mining
script_dir = os.path.dirname(__file__)
file_to_learn = os.path.join(script_dir, '../autofunc/assets/reservoir_systems.csv')

# CSV with systems to test (blade, heating element, reservoir, etc.)
file_to_test = os.path.join(script_dir, '../autofunc/assets/reservoir_systems.csv')

test_data = pd.read_csv(file_to_test)
train_data = pd.read_csv(file_to_learn)

# test_ids = list(store_data.id.unique())
test_ids = list(map(int,test_data.id.unique()))
train_ids = list(map(int,train_data.id.unique()))
random.shuffle(test_ids)


# Use a threshold to get the top XX% of confidence values
threshold = 0.7

# Pandas
df = make_df(file_to_learn)

# # Split into folds
k = 10

keep = []
plots = []
precisions = []
recalls = []

f1s = 0


## If using B&D
# BD Dataframe
if bd:
    bd_file = os.path.join(script_dir, '../autofunc/assets/bd_systems.csv')
    bd_df = make_df(bd_file)
    bd_ids = list(map(int, bd_df.id.unique()))

    new_ids = [x for x in test_ids if x not in bd_ids]

    n = floor(len(new_ids)/k)
    folds = [new_ids[i * n:(i + 1) * n] for i in range((len(new_ids) + n - 1) // n)]


iters = 10
count = 0

averages = []

for i in range(iters):
    random.shuffle(test_ids)
    # Split into folds
    n = floor(len(test_ids) / k)
    # Making folds using list comprehension
    folds = [test_ids[i * n:(i + 1) * n] for i in range((len(test_ids) + n - 1) // n)]

    for e in folds:
        verification_ids = e

        ver_df, learn_df = split_learning_verification(df, verification_ids)

        ver_list = df_to_list(ver_df)

        if not bd:
            comb_sort, counts, combos = counter_pandas_with_counts(learn_df)
            thresh_results = get_top_results(comb_sort, threshold)

            # Find the F1 score of the verification test by comparing the learned results with the known function/flows
            learned_dict, matched, overmatched, unmatched, recall, precision, f1 = precision_recall(thresh_results, ver_list)


        if bd:
            bd_comb_sort = counter_pandas(bd_df)
            bd_thresh_results = get_top_results(bd_comb_sort, threshold)
            learned_dict, matched, overmatched, unmatched, recall, precision, f1 = precision_recall(bd_thresh_results, ver_list)

        precisions.append(precision)
        recalls.append(recall)

        print(e)

        f1s += f1

        keep.append([e, f1])
        plots.append(f1)

        avg_f1 = f1s / len(keep)

        count += 1

    print(avg_f1)
    averages.append(avg_f1)
    avg_f1 = 0
    count = 0



optimum = max(keep,key=itemgetter(1))
avg_f1 = mean(averages)

print('Maximum is {0:.2f}'.format(optimum[1]))
print('Average F1 is {0:.4f}'.format(avg_f1))


end = time.time()
print('Time is {0:.2f}'.format(end - start))