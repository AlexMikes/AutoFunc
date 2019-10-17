from autofunc.get_match_factor import match
from autofunc.get_precision_recall import precision_recall
from autofunc.simple_counter import count_stuff
from autofunc.get_top_results import get_top_results
from autofunc.get_data import get_data
from autofunc.counter_pandas import counter_pandas
from autofunc.counter_pandas_with_counts import counter_pandas_with_counts
from autofunc.make_df import make_df
from autofunc.split_learning_verification import split_learning_verification
from autofunc.df_to_list import df_to_list
from autofunc.write_results import write_results_from_dict
from autofunc.find_associations import find_associations
import os.path
from statistics import mean
from statistics import harmonic_mean as hm
import pandas as pd


""" Example showing how to find the match factor using the simple counting file """

script_dir = os.path.dirname(__file__)
file1 = os.path.join(script_dir, '../assets/blade_systems.csv')

# Pandas
df = make_df(file1)


comb_sort, counts, combos = counter_pandas_with_counts(df)

comb_sort2 = counter_pandas(df)

threshold = 0.7
thresh_results = get_top_results(comb_sort2, threshold)

# Scaling by maximum number of each component
max_component = counts[max(counts, key=counts.get)]
max_component_name = max(counts, key=counts.get)

average = mean(counts[k] for k in counts)


average_counter = 0
average_combos = 0
total_cffs = 0
average_count = {}
count_cffs = {}

for k,v in combos.items():
    for k2,v2 in combos[k].items():
        average_combos += v2
        average_counter += 1

    count_cffs[k] = len(combos[k])
    average_count[k] = average_combos/average_counter
    total_cffs += average_combos
    average_counter = 0
    average_combos = 0



scaled = {}
scaled_conf = {}
averaged = {}
averaged_combos = {}
combined = {}


for k,v in counts.items():
    scaled[k] = v/max_component
    averaged[k] = v/average
    combined[k] = [counts[k],averaged[k],scaled[k],count_cffs[k], average_count[k]]


# combined_thresh_results = thresh_results
#
# for k,v in combined_thresh_results.items():
#     for vs in v:
#         for es in combined[k]:
#             vs.append(es)



# Columns to use: Normalized average{}, Normalized average_count[k]
# Normalized average tells how popular the component is compared to the average
# Average count shows the comparison to the average variance in unique CFFs


combined_thresh_results = thresh_results



max_averaged = averaged[max(averaged, key=averaged.get)]
max_average_count = average_count[max(average_count, key=average_count.get)]

for k,v in combined_thresh_results.items():
    for vs in v:
        for es in combined[k]:
            vs.append(es)
        av_norm = averaged[k] / max_averaged
        av_count_norm = average_count[k]/max_average_count
        ratio_unique = len(thresh_results[k]) / count_cffs[k]
        in_thresh = len(thresh_results[k])
        vs.extend([av_norm, av_count_norm, in_thresh, ratio_unique, hm((av_norm,av_count_norm))])






# Writing results with stuff from Rob meeting

titles = ['comp','func-flow','prob','Total CFF counts', 'averaged(/{0})'.format(average),
          'scaled(/{0})'.format(max_component), '# of unique CFFs','average # CFFs','normalized occurrences',
          'normalized unique CFFs','# unique CFFs in thresh','ratio unique','harmonic mean']



write_results_from_dict(combined_thresh_results, 'blade_counts5.csv',titles)





# test = pd.DataFrame.from_dict(combined, orient='index', columns=['Total CFF counts', 'averaged(/{0})'.format(average), 'scaled(/{0})'.format(max_component), '# of unique CFFs','average # CFFs'])

# test.to_csv('reservoir_counts.csv', index=True)




# for k,v in thresh_results.items():
#
#     scaled_conf[k][v] = thresh_results[k][v]*(v/max_component)







