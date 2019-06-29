from autofunc.get_percent_similar import *
from autofunc.get_match_factor import match
from autofunc.simple_counter import count_stuff
from autofunc.count_with_id import *
from autofunc.get_top_results import get_top_results
from autofunc.get_data import get_data
import os.path
import matplotlib.pyplot as plt

# Dataset used for data mining
script_dir = os.path.dirname(__file__)
file1 = os.path.join(script_dir, '../assets/bladeCombined_id.csv')
file2 = os.path.join(script_dir, '../assets/jigsawQuery_headers.csv')

input_data = pd.read_csv(file1)

# ids = list(store_data.id.unique())
ids = list(map(int,input_data.id.unique()))

# test_id = ids[3]

# learning_comps = test_data.loc[test_data['id'] == test_id]['comp']
#
# unq_learning_comps = list(learning_comps.unique())

ps_thresh = []
matches = []
keep_ps = []
keep_ps_thresh = []

threshold = 0.5

for i in range(0,100,10):

    keep_ids = []

    ps_thresh = i/100

    for id in ids:

        ps = percent_similar(file1,file2, id)

        print(ps)

        if ps > ps_thresh:

            keep_ids.append(id)
            keep_ps.append(ps)

    # Only keep rows from data frame that have an id that is in the keep_ids list
    keep_df = input_data[input_data['id'].isin(keep_ids)]

    # Name each file for writing then reading back in
    s = ['../opt/', str(ps_thresh),'.csv']

    sep = ''

    name = sep.join(s)

    # Write each file with the name of the threshold
    export_csv = keep_df.to_csv(os.path.join(script_dir, name), index = None, header=True)



    # Re-analyze by reading each file in
    file3 = os.path.join(script_dir, name)

    comb_sort = count_stuff(file3)

    thresh_results = get_top_results(comb_sort, threshold)

    # Use a known product for verification

    test_data, test_records = get_data(file2)

    # Find the match factor of the verification test by comparing the learned results with the known function/flows
    learned_dict, matched, overmatched, unmatched, match_factor = match(thresh_results, test_records)

    matches.append(match_factor)
    keep_ps_thresh.append(ps_thresh)




# Find max match factor and corresponding threshold
m = max(matches)
ind = matches.index(m)

opt = keep_ps_thresh[ind]

print('Optimum Threshold = {0:.5f}'.format(opt))

plt.plot(keep_ps_thresh, matches)
plt.xlabel('Percent Similar Threshold')
plt.ylabel('Match Factor')
plt.title('Match Factor vs Percent Similar Threshold')
plt.grid()
plt.show()


# Getting match factors, comparing with percent similar


