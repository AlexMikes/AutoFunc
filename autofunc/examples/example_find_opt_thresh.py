from autofunc.get_match_factor import match
from autofunc.simple_counter import count_stuff
from autofunc.get_top_results import get_top_results
from autofunc.get_data import get_data
import os.path
import matplotlib.pyplot as plt

""" Example showing how to find the optimum threshold based on the resulting match factor  """


# Dataset used for data mining
script_dir = os.path.dirname(__file__)
# file1 = os.path.join(script_dir, '../assets/bladeCombined.csv')
file1 = os.path.join(script_dir, '../assets/all_products_id_autofunc.csv')

comb_sort = count_stuff(file1)

threshes = []
matches = []

for i in range(10,100,5):

    threshold = i/100

    thresh_results = get_top_results(comb_sort, threshold)

    # Use a known product for verification
    test_file = os.path.join(script_dir, '../assets/jigsawQuery.csv')

    test_data, test_records = get_data(test_file)

    # Find the match factor of the verification test by comparing the learned results with the known function/flows
    learned_dict, matched, overmatched, unmatched, match_factor = match(thresh_results, test_records)

    threshes.append(threshold)
    matches.append(match_factor)

# Find max match factor and corresponding threshold
m = max(matches)
ind = matches.index(m)

opt = threshes[ind]

print('Optimum Threshold = {0:.5f}'.format(opt))

plt.plot(threshes, matches)
plt.xlabel('Threshold')
plt.ylabel('Match Factor')
plt.title('Match Factor vs Threshold')
plt.grid()
plt.show()