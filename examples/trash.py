"""

Example showing how to implement all of the main modules and optimize the similarity and classification thresholds

It performs a leave-one-out cross-validation in a triple nested loop:

1. Iterate through which product is the testing set
    2. Iterate through similarity thresholds and make training sets based on which products are within the threshold
        3. Iterate through classification thresholds and find the F1 scores for each combination of testing product,
        similarity threshold, and classification threshold

The optimization process is simply finding the highest average F1 score and extracting the similarity and classification
thresholds for that number.

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
file_to_learn = os.path.join(script_dir, '../assets/consumer_systems.csv')

train_data = pd.read_csv(file_to_learn)
train_df_whole = make_df(file_to_learn)

all_train_ids = list(map(int, train_data.id.unique()))

ps_thresh = []
f1s = []
keep_ps = []
keep_ps_thresh = []
threshes = []
ps_time = []
comp_ratios = []
points = []
save_data = []
combinations_used = []

## Uncomment 1 or 2, not both
## If generating similarity dataframe for the first time, uncomment 1 and comment 2
## If reading in the similarity dataframe from 1, uncomment 2 and comment 1
## Reading boolean changes datatype to string when finding keep_ids below to deal with how
## Pandas reads in a csv to a dataframe being different than how it makes its own the first time

## 1. Make similarity dataframe, takes a while so it is saved to csv first for reading in later
# similarity_df = find_similarities(train_data)
# similarity_df.to_csv('consumer_similarity.csv', index = True, index_label=False, header= True)
# reading = False
## End 1

## 2. Reading in dataframe as computed above
similarity_df = pd.read_csv('consumer_similarity.csv')
reading = True
## End 2


all_comps = list(train_df_whole.comp.unique())
num_all_comps = len(all_comps)

for test_id in all_train_ids:
    # print(test_id)
    #
    test_df, train_df = split_learning_verification(train_df_whole, [test_id])
    test_list = df_to_list(test_df)
    train_ids = list(map(int, train_df.id.unique()))
    #
    # Outer loop through percent similar
    for i in range(0, 100, 10):

        ps_start = time.time()

        f1_plot = []
        thresh_plot = []
        ps_plot = []

        keep_ids = []

        ps_thresh = i / 100

        if reading:
            keep_ids = similarity_df[similarity_df[str(test_id)] > ps_thresh].index.tolist()
        else:
            keep_ids = similarity_df[similarity_df[test_id] > ps_thresh].index.tolist()

        keep_ids.remove(test_id)

        # Only keep rows from data frame that have an id that is in the keep_ids list
        keep_df = train_df[train_df['id'].isin(keep_ids)]

        comb_sort = counter_pandas(keep_df)

        # Component counting and fractions
        train_comps = list(keep_df.comp.unique())

        if train_comps:
            comp_ratio = len(train_comps) / num_all_comps
            comp_ratios.append((len(keep_ids), i, len(train_comps) / num_all_comps))

        if comp_ratio > 0.7 and len(keep_ids) < 40:

            for t in range(10, 100, 5):
                threshold = t / 100
                print(test_id, ' ', ps_thresh, ' ', threshold)

                thresh_results = get_top_results(comb_sort, threshold)

                if not keep_ids:
                    f1 = 0
                    num_train_comps = 0
                else:
                    # Find the F1 score of the verification test by comparing the learned results with the known function/flows
                    learned_dict, matched, overmatched, unmatched, recall, precision, f1 = precision_recall(
                        thresh_results,
                        test_list)
                    num_train_comps = len(train_comps)

                if (i, t) not in combinations_used:
                    combinations_used.append((i, t))

                save_data.append((test_id, ps_thresh, threshold, len(keep_ids), f1, num_train_comps / num_all_comps))

                points.append((ps_thresh, threshold, f1))

                f1s.append(f1)
                # keep_ps_thresh.append(ps_thresh)
                # threshes.append(threshold)
                #
                # f1_plot.append(f1)
                # thresh_plot.append(threshold)
                # ps_plot.append(ps_thresh)
                #

        else:
            f1 = 0
            # save_data.append((test_id, ps_thresh, threshold, len(keep_ids), f1, num_train_comps / num_all_comps))

        ps_end = time.time()
        ps_time.append((len(keep_ids), (ps_end - ps_start)))

# MultiIndex if needed
# index = pd.MultiIndex.from_tuples(save_data, names=['Product ID', 'PS Thresh','Thresh','Num Keep IDs'])
# all_data = pd.Series(f1s, index=index )

all_data = pd.DataFrame(save_data,
                        columns=['Test Product ID', 'PS Thresh', 'Thresh', 'Num Keep IDs', 'F1', 'Comp Ratio'])

# Uncomment to write to csv
# all_data.to_csv('consumer_opt_export.csv', index = False, header=True)

averages = []
average_ids = []
avg_avgf1 = []
optimums = []

ids_3d = []
ps_3d = []
thresh_3d = []
f1_3d = []
ids_plot = []
ps_plot = []
comp_ratio_plot = []

# for i in range(0, 100, 10):
for e in combinations_used:
    i = e[0]
    t = e[1]
    ps_thresh = i / 100

    opt_finder = []
    f1_plot = []
    thresh_plot = []

    try:
        avg_ids = mean(
            all_data['Num Keep IDs'][(all_data['Thresh'] == threshold) & (all_data['PS Thresh'] == ps_thresh)])
        average_ids.append((ps_thresh, threshold, avg_ids))
        ids_plot.append(avg_ids)
    except:
        print('No matching products for this combination')

    ps_plot.append(ps_thresh)

    try:
        avg_comp_ratio = mean(
            all_data['Comp Ratio'][(all_data['Thresh'] == threshold) & (all_data['PS Thresh'] == ps_thresh)])
        comp_ratio_plot.append(avg_comp_ratio)
    except:
        print('No matching ratios for this combination')

    # for t in range(10, 100, 5):
    threshold = t / 100

    # Find the average F1 score for each combination of threshold and percent similar
    try:
        avg_f1 = mean(all_data['F1'][(all_data['Thresh'] == threshold) & (all_data['PS Thresh'] == ps_thresh)])
        averages.append((ps_thresh, threshold, avg_f1, avg_ids))
        opt_finder.append((ps_thresh, threshold, avg_f1, avg_ids))
        f1_plot.append(avg_f1)
    except:
        print('No F1s for this combinations')

    thresh_plot.append(threshold)

    ids_3d.append(avg_ids)
    ps_3d.append(ps_thresh)
    thresh_3d.append(threshold)
    f1_3d.append(avg_f1)

    # Find best threshold for each percent similar
    optimums.append(max(opt_finder, key=itemgetter(2)))

    avg_avgf1.append(mean(f1_plot))

avg_opt = mean([x[1] for x in optimums])
optimum = max(averages, key=itemgetter(2))
print('Optimum Percent Similar Threshold = {0:.2f}'.format(optimum[0]))
print('Optimum Threshold = {0:.2f}'.format(optimum[1]))
print('Maximum F1 = {0:.4f}'.format(optimum[2]))

# # Plotting f1s vs ps at 0.55 threshold
# plot_f1s = [x[2] for x in averages if x[1] == 0.55]
# plt.plot(ps_plot, plot_f1s)
# plt.xlabel('Similarity Threshold')
# plt.ylabel('Average F1 Score')
# # plt.title('F1 Score vs PS at 0.55 threshold')
# plt.grid()
# plt.show()

# # Plotting f1s vs threshold at 0.2 ps
# plot_f1s = [x[2] for x in averages if x[0] == 0.2]
# plt.plot(thresh_plot, plot_f1s)
# plt.xlabel('Classification Threshold')
# plt.ylabel('Average F1 Score')
# # plt.title('F1 Score vs Threshold at 0.2 Percent Similar')
# plt.grid()
# plt.show()

# Plotting f1 vs num ids
plt.plot(ps_plot, ids_plot)
plt.xlabel('Similarity Threshold')
plt.ylabel('Number of Products in Training Set')
# plt.title('Number of products vs similarity threshold')
plt.grid()
plt.show()

# Plotting f1 vs num ids
plt.plot(ids_plot, avg_avgf1)
plt.ylabel('Average F1 Score')
plt.xlabel('Number of Products in Training Set')
# plt.title('Avg F1 score vs Number of Products')
plt.grid()
plt.show()

# Plotting f1 vs comp ratio
plt.plot(comp_ratio_plot, avg_avgf1)
plt.ylabel('Average F1 Score')
plt.xlabel('Average Component Ratios')
# plt.title('Avg F1 score vs Component Ratios')
plt.grid()
plt.show()

# # Plotting comp ratio and num prods vs f1
# x = avg_avgf1
# y1 = comp_ratio_plot
# y2 = ids_plot
# fig, ax1 = plt.subplots()
# color = 'tab:red'
# ax1.set_xlabel('time (s)')
# ax1.set_ylabel('exp', color=color)
# ax1.plot(x, y1, color=color)
# ax1.tick_params(axis='y', labelcolor=color)
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# color = 'tab:blue'
# ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
# ax2.plot(x, y2, color=color)
# ax2.tick_params(axis='y', labelcolor=color)
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()


# Plotting comp ratio and f1 vs num proudcts in training set
x = ids_plot
y1 = comp_ratio_plot
y2 = avg_avgf1
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Number of Products in Training Set')
ax1.set_ylabel('Ratio of Components Represented', color=color)
ax1.plot(x, y1, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Average F1 scores', color=color)  # we already handled the x-label with ax1
ax2.plot(x, y2, color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.axvline(x=27)
plt.grid()
plt.show()

# Plot comp ratio vs num prods
plt.plot(ids_plot, comp_ratio_plot)
plt.xlabel('Average Number of Proucts in Training Set')
plt.ylabel('Average Component Ratios')
# plt.title('Avg F1 score vs Number of Products')
plt.grid()
plt.show()

# Plotting # Prods vs time
plot_time = [x[1] for x in ps_time]
plot_num_ids = [x[0] for x in ps_time]
plot_time.sort()
plot_num_ids.sort()
plt.plot(plot_time, plot_num_ids)
plt.xlabel('Time (s)')
plt.ylabel('Number of Products in Training Set')
# plt.title('Number of Products vs. Time ')
plt.grid()
plt.show()

# 3D Scatter Plot
ax = plt.axes(projection='3d')

zdata = f1_3d
ydata2 = ids_3d
xdata = ps_3d
ydata = thresh_3d

# Data for three-dimensional scattered points of F1
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Dark2');
ax.set_xlabel('Similarity Threshold')
ax.set_ylabel('Classification Threshold')
ax.set_zlabel('Average F1 Score');
plt.show()

# ## 3D Scatter plot of percent similar, num ids, f1: Kind of weird
# # Data for three-dimensional scattered points of ids
# ax2 = plt.axes(projection='3d')
# ax2.scatter3D(xdata, ydata2, zdata, c=zdata, cmap='Dark2');
# ax2.set_xlabel('Percent Similar')
# ax2.set_ylabel('Num IDs')
# ax2.set_zlabel('F1 Score');
# plt.show()


# Optional write all data to a dataframe and csv file
# avg_df = pd.DataFrame(averages,columns = ['PS Thresh','Thresh','Avg F1', 'Avg IDs'])
# avg_df.to_csv('averages.csv',index = False, header=True)

end = time.time()
print('Time is {0:.2f}'.format(end - start))

# #3D Surface Plot
# X, Y, Z = np.meshgrid(xdata, ydata, zdata)
# fig = plt.figure()
# # ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50, cmap='binary')



