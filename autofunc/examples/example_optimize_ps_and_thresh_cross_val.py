from autofunc.get_percent_similar import *
from autofunc.get_precision_recall import precision_recall
from autofunc.get_top_results import get_top_results
from autofunc.make_df import make_df
from autofunc.counter_pandas import counter_pandas
from autofunc.split_learning_verification import split_learning_verification
from autofunc.df_to_list import df_to_list
import os.path
import matplotlib.pyplot as plt
from operator import itemgetter
import numpy as np
from mpl_toolkits import mplot3d
import time



## New algorithm
# 1. Pull out test set (one product) from training set
# 2. Make dfs
# 3. Loop percent similar
# 4. Loop threshold
# 5.


start = time.time()

# Dataset used for data mining
script_dir = os.path.dirname(__file__)
file_to_learn = os.path.join(script_dir, '../assets/blade_systems.csv')

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

# threshold = 0.5

points = []


for test_id in all_train_ids:

    test_df, train_df = split_learning_verification(train_df_whole, [test_id])
    test_list = df_to_list(test_df)
    train_ids = list(map(int, train_df.id.unique()))

    for i in range(0,100,10):

        f1_plot = []
        thresh_plot = []
        ps_plot = []

        keep_ids = []

        ps_thresh = i/100

        for train_id in train_ids:

            ps = percent_similar(train_df, test_df, train_id)

            print(ps)

            if ps > ps_thresh:

                keep_ids.append(test_id)
                keep_ps.append(ps)

        # Only keep rows from data frame that have an id that is in the keep_ids list
        keep_df = train_data[train_data['id'].isin(keep_ids)]

        # # Name each file for writing then reading back in
        # s = ['../opt/', str(ps_thresh),'.csv']
        # sep = ''
        # name = sep.join(s)
        #
        # # Write each file with the name of the threshold
        # export_csv = keep_df.to_csv(os.path.join(script_dir, name), index = None, header=True)
        #
        #
        #
        # # Re-analyze by reading each file in
        # file3 = os.path.join(script_dir, name)

        # comb_sort = count_stuff(file3)

        comb_sort = counter_pandas(keep_df)


        ## End Not B&D


        for t in range(10, 100, 5):
            threshold = t / 100

            # thresh_results = get_top_results(comb_sort, threshold)

            # Use a known product for verification

            # test_data, test_records = get_data(file2)

            # Find the match factor of the verification test by comparing the learned results with the known function/flows
            # learned_dict, matched, overmatched, unmatched, match_factor = match(thresh_results, test_records)

            # learned_dict, matched, overmatched, unmatched, recall, precision, f1 = precision_recall(thresh_results,test_records)

            thresh_results = get_top_results(comb_sort, threshold)

            if not keep_ids:
                f1 = 0
            else:
                # Find the F1 score of the verification test by comparing the learned results with the known function/flows
                learned_dict, matched, overmatched, unmatched, recall, precision, f1 = precision_recall(thresh_results,
                                                                                                    test_list)


            points.append((ps_thresh,threshold,f1))


            f1s.append(f1)
            keep_ps_thresh.append(ps_thresh)
            threshes.append(threshold)

            f1_plot.append(f1)
            thresh_plot.append(threshold)
            ps_plot.append(ps_thresh)



        #Plotting in loop for each threshold
        plt.plot(thresh_plot, f1_plot)
        plt.xlabel('Threshold')
        plt.ylabel('F1')
        plt.title('PS = {0:.2f}'.format(ps_thresh))
        plt.grid()
        plt.show()



# Find the tuple with the highest match factor
optimum = max(points,key=itemgetter(2))

print('Optimum Percent Similar Threshold = {0:.2f}'.format(optimum[0]))
print('Optimum Threshold = {0:.2f}'.format(optimum[1]))
print('Maximum F1 = {0:.4f}'.format(optimum[2]))

end = time.time()
print('Time is {0:.2f}'.format(end - start))


ax = plt.axes(projection='3d')


zdata = f1s
xdata = keep_ps_thresh
ydata = threshes


# 3D Scatter Plot
# Data for three-dimensional scattered points
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Dark2');
ax.set_xlabel('Percent Similar')
ax.set_ylabel('Threshold')
ax.set_zlabel('F1 Score');
plt.show()
#
#
#
# #3D Surface Plot
# X, Y, Z = np.meshgrid(xdata, ydata, zdata)
# fig = plt.figure()
# # ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50, cmap='binary')




# # Find max match factor and corresponding threshold
# m = max(matches)
# ind = matches.index(m)
#
# opt_ps = keep_ps_thresh[ind]
#
# print('Optimum Percent Similar = {0:.5f}'.format(opt_ps))
#
# # Find max match factor and corresponding threshold
# m = max(matches)
# ind = matches.index(m)
#
# opt = threshes[ind]
#
# print('Optimum Threshold = {0:.5f}'.format(opt))
#
# plt.plot(keep_ps_thresh, matches)
# plt.xlabel('Percent Similar Threshold')
# plt.ylabel('Match Factor')
# plt.title('Match Factor vs Percent Similar Threshold')
# plt.grid()
# plt.show()


# Getting match factors, comparing with percent similar


