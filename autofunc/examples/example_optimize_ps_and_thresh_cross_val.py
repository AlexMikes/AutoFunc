from autofunc.get_percent_similar import *
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



## New algorithm
# 1. Pull out test set (one product) from training set
# 2. Make dfs
# 3. Loop percent similar
# 4. Loop threshold
# 5.


start = time.time()

# Dataset used for data mining
script_dir = os.path.dirname(__file__)
file_to_learn = os.path.join(script_dir, '../assets/consumer_systems.csv')

train_data = pd.read_csv(file_to_learn)
train_df_whole = make_df(file_to_learn)


# ids = list(store_data.id.unique())
# ids = list(map(int,input_data.id.unique()))
all_train_ids = list(map(int,train_data.id.unique()))

# test_id = ids[3]

# learning_comps = test_data.loc[test_data['id'] == test_id]['comp']
#
# unq_learning_comps = list(learning_comps.unique())

ps_thresh = []
f1s = []
keep_ps = []
keep_ps_thresh = []
threshes = []
ps_time = []

points = []
save_data = []


## Uncomment 1 or 2, not both
## Reading boolean changes datatype to string when finding keep_ids below to deal with how
## Pandas reads in a csv to dataframe being different than how it makes its own the first time
# 1. Make similarity dataframe, takes a while so it is saved to csv first for reading in later
# similarity_df = find_similarities(train_data)
# similarity_df.to_csv('blade_similarity.csv', index = True, index_label=False, header= True)
# reading = False

# 2. Reading in dataframe as computed above
similarity_df = pd.read_csv('consumer_similarity.csv')
reading = True




# greater_than_threshold = similarity_df[similarity_df[169]>0.5].index.tolist()


for test_id in all_train_ids:
    # print(test_id)
#
    test_df, train_df = split_learning_verification(train_df_whole, [test_id])
    test_list = df_to_list(test_df)
    train_ids = list(map(int, train_df.id.unique()))

#
# Outer loop through percent similar
    for i in range(0,100,10):

        ps_start = time.time()

        f1_plot = []
        thresh_plot = []
        ps_plot = []

        keep_ids = []

        ps_thresh = i/100

        if reading:
            keep_ids = similarity_df[similarity_df[str(test_id)] > ps_thresh].index.tolist()
        else:
            keep_ids = similarity_df[similarity_df[test_id] > ps_thresh].index.tolist()


        keep_ids.remove(test_id)


        # Only keep rows from data frame that have an id that is in the keep_ids list
        keep_df = train_df[train_df['id'].isin(keep_ids)]

        comb_sort = counter_pandas(keep_df)


        for t in range(10, 100, 5):
            threshold = t / 100
            print(test_id, ' ', ps_thresh, ' ',threshold)

            thresh_results = get_top_results(comb_sort, threshold)

            if not keep_ids:
                f1 = 0
            else:
                # Find the F1 score of the verification test by comparing the learned results with the known function/flows
                learned_dict, matched, overmatched, unmatched, recall, precision, f1 = precision_recall(thresh_results,
                                                                                                    test_list)


            save_data.append((test_id,ps_thresh,threshold, len(keep_ids),f1))

            points.append((ps_thresh,threshold,f1))

            f1s.append(f1)
            # keep_ps_thresh.append(ps_thresh)
            # threshes.append(threshold)
            #
            # f1_plot.append(f1)
            # thresh_plot.append(threshold)
            # ps_plot.append(ps_thresh)
            #
        ps_end = time.time()
        ps_time.append((len(keep_ids), (ps_end - ps_start)))

# MultiIndex if needed
# index = pd.MultiIndex.from_tuples(save_data, names=['Product ID', 'PS Thresh','Thresh','Num Keep IDs'])
# all_data = pd.Series(f1s, index=index )

all_data = pd.DataFrame(save_data,columns = ['Product ID', 'PS Thresh','Thresh','Num Keep IDs','F1'])

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

for i in range(0,100,10):
    ps_thresh = i / 100

    opt_finder = []
    f1_plot = []
    thresh_plot = []


    avg_ids = mean(all_data['Num Keep IDs'][(all_data['Thresh'] == threshold) & (all_data['PS Thresh'] == ps_thresh)])
    average_ids.append((ps_thresh, threshold, avg_ids))
    ids_plot.append(avg_ids)
    ps_plot.append(ps_thresh)

    for t in range(10, 100, 5):
        threshold = t / 100

        # Find the average F1 score for each combination of threshold and percent similar
        avg_f1 = mean(all_data['F1'][(all_data['Thresh'] == threshold) & (all_data['PS Thresh'] == ps_thresh)])
        averages.append((ps_thresh,threshold,avg_f1, avg_ids))
        opt_finder.append((ps_thresh,threshold,avg_f1, avg_ids))

        f1_plot.append(avg_f1)
        thresh_plot.append(threshold)


        ids_3d.append(avg_ids)
        ps_3d.append(ps_thresh)
        thresh_3d.append(threshold)
        f1_3d.append(avg_f1)

    # Find best threshold for each percent similar
    optimums.append(max(opt_finder,key=itemgetter(2)))

    avg_avgf1.append(mean(f1_plot))


    # Line plot of f1 vs threshold for each percent similar
    # #Plotting in loop for each threshold
    # plt.plot(thresh_plot, f1_plot)
    # plt.xlabel('Threshold')
    # plt.ylabel('F1')
    # plt.title('PS = {0:.2f}'.format(ps_thresh))
    # plt.grid()
    # plt.show()

avg_opt = mean([x[1] for x in optimums])
optimum = max(averages,key=itemgetter(2))
print('Optimum Percent Similar Threshold = {0:.2f}'.format(optimum[0]))
print('Optimum Threshold = {0:.2f}'.format(optimum[1]))
print('Maximum F1 = {0:.4f}'.format(optimum[2]))

# Plotting f1s vs ps at 0.55 threshold
plot_f1s = [x[2] for x in averages if x[1] == 0.55]
plt.plot(ps_plot, plot_f1s)
plt.xlabel('Similarity Threshold')
plt.ylabel('F1 Score')
# plt.title('F1 Score vs PS at 0.55 threshold')
plt.grid()
plt.show()

# Plotting f1s vs threshold at 0.2 ps
plot_f1s = [x[2] for x in averages if x[0] == 0.2]
plt.plot(thresh_plot, plot_f1s)
plt.xlabel('Classification Threshold')
plt.ylabel('F1 Score')
# plt.title('F1 Score vs Threshold at 0.2 Percent Similar')
plt.grid()
plt.show()

#Plotting f1 vs num ids
plt.plot(ps_plot, ids_plot)
plt.xlabel('Similarity Threshold')
plt.ylabel('Number of Products in Training Set')
# plt.title('Number of products vs similarity threshold')
plt.grid()
plt.show()

#Plotting f1 vs num ids
plt.plot(ids_plot,avg_avgf1)
plt.ylabel('Average F1 Score')
plt.xlabel('Number of Products in Training Set')
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
ax.set_zlabel('F1 Score');
plt.show()


# ## 3D Scatter plot of percent similar, num ids, f1: Kind of weird
# # Data for three-dimensional scattered points of ids
# ax2 = plt.axes(projection='3d')
# ax2.scatter3D(xdata, ydata2, zdata, c=zdata, cmap='Dark2');
# ax2.set_xlabel('Percent Similar')
# ax2.set_ylabel('Num IDs')
# ax2.set_zlabel('F1 Score');
# plt.show()



avg_df = pd.DataFrame(averages,columns = ['PS Thresh','Thresh','Avg F1', 'Avg IDs'])

# avg_df.to_csv('averages.csv',index = False, header=True)


# # def is_pareto_efficient_simple(costs):
# #     """
# #     Find the pareto-efficient points
# #     :param costs: An (n_points, n_costs) array
# #     :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
# #     """
# is_efficient = np.ones(points.shape[0], dtype = bool)
# for i, c in enumerate(costs):
#     if is_efficient[i]:
#         is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
#         is_efficient[i] = True  # And keep self
#     # return is_efficient

end = time.time()
print('Time is {0:.2f}'.format(end - start))

# #3D Surface Plot
# X, Y, Z = np.meshgrid(xdata, ydata, zdata)
# fig = plt.figure()
# # ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50, cmap='binary')



