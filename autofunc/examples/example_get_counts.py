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
import os.path
from statistics import mean
from statistics import harmonic_mean as hm
import pandas as pd
import copy


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

# for k,v in combos.items():
#     for k2,v2 in combos[k].items():
#         average_combos += v2
#         average_counter += 1
#
#     count_cffs[k] = len(combos[k])
#     average_count[k] = average_combos/average_counter
#     total_cffs += average_combos
#     average_counter = 0
#     average_combos = 0



scaled = {}
scaled_conf = {}
averaged = {}
averaged_combos = {}
combined = {}


# for k,v in counts.items():
#     scaled[k] = v/max_component
#     averaged[k] = v/average
#     combined[k] = [counts[k],averaged[k],scaled[k],count_cffs[k], average_count[k]]


# combined_thresh_results = thresh_results
#
# for k,v in combined_thresh_results.items():
#     for vs in v:
#         for es in combined[k]:
#             vs.append(es)



# Columns to use: Normalized average{}, Normalized average_count[k]
# Normalized average tells how popular the component is compared to the average
# Average count shows the comparison to the average variance in unique CFFs


combined_thresh_results = copy.deepcopy(thresh_results)

# Old with too much output

# max_averaged = averaged[max(averaged, key=averaged.get)]
# max_average_count = average_count[max(average_count, key=average_count.get)]
#
# for k,v in combined_thresh_results.items():
#     for vs in v:
#         vs.append(combos[k][vs[0]])
#         vs.append(len(df))
#         for es in combined[k]:
#             vs.append(es)
#
#         av_norm = averaged[k] / max_averaged
#         av_count_norm = average_count[k]/max_average_count
#         ratio_unique = len(thresh_results[k]) / count_cffs[k]
#         in_thresh = len(thresh_results[k])
#         vs.extend([av_norm, av_count_norm, in_thresh, ratio_unique, hm((av_norm,av_count_norm))])


# New with refined output

scaled = {}

for k,v in counts.items():
    scaled[k] = v/max_component


count_thresh_cffs = {}
cff_counter = 0
total_counter = 0



for k,v in combined_thresh_results.items():
    for vs in v:
        cff_counter += combos[k][vs[0]]  # Sum number of CFF occurrences in threshold

    count_thresh_cffs[k] = cff_counter  # Store the sum for each component


    count_cffs[k] = len(combos[k])  # Total number of CFFs not only in threshold
    total_counter += cff_counter    # Store total CFFs in threshold
    cff_counter = 0




hm_scaled = {}
hm_ratio = {}
ratio_uniques = {}
norm_ratio = {}

for k,v in combined_thresh_results.items():
    for vs in v:
        ratio_uniques[k] = count_cffs[k] / len(thresh_results[k])  #Total unique CFFs / Unique CFFs in threshold
        # cffs_in_thresh[k] += len(v)

max_ratio_uniques = ratio_uniques[max(ratio_uniques, key=ratio_uniques.get)]

for k,v in combined_thresh_results.items():
    norm_ratio[k] = ratio_uniques[k] / max_ratio_uniques
    scaled_num = scaled[k]

    hm_scaled[k] = hm((scaled_num,norm_ratio[k]))


    for vs in v:
        vs.append(combos[k][vs[0]])  # Number of CFF occurrences for each CFF
        vs.append(count_thresh_cffs[k])  # Sum of individual CFFS in threshold for each component
        vs.append(counts[k])  # Total CFFs for each component
        vs.append(count_cffs[k]) # Number of unique CFFs for each component
        vs.append(len(thresh_results[k]))  # Number of unique CFFs in threshold
        prob = vs[1]
        hm_ratio[k] = hm((prob, scaled_num, norm_ratio[k]))
        vs.extend([scaled_num, norm_ratio[k], hm_scaled[k],
                   hm_ratio[k]])

titles = ['comp', 'func-flow', 'prob', '# CFF occurrences', 'Sum of CFFs in threshold',  'Total CFF counts',
          'Number unique CFFs', '# Unique CFFs in thresh', 'scaled CFFs (#CFFs/max)',
          'Normalized ratio uniques (norm(total unique CFFs/unique CFFS in thresh))',
          'hm scaled, ratio uniques captured', 'hm scaled, ratio uniques captured, probability']

## 10/25 add: normalized(count_cffs[k]/len(thresh_results[k]))   ( This is column P)
## Harmonic mean with scaled
## Harmonic mean with probability



## Plotting
import matplotlib.pyplot as plt

## Scatter plot
points = []
points_dict = {}
xs = []
ys = []
for k,v in combined_thresh_results.items():
    points_dict[k] = (hm_scaled[k], hm_ratio[k])
    points.append((hm_scaled[k], hm_ratio[k]))
    xs.append(scaled[k])
    ys.append(norm_ratio[k])


# area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

plt.plot(xs, ys, 'o',alpha=0.5)
plt.xlim(0, 1.2)
plt.ylim(0, 1.2)
plt.xlabel('Prevalence')
plt.ylabel('Consistency')
plt.show()


## Bar Chart


# plt.style.use('ggplot')

xs_bar = []
ys_bar = []

k = 'battery'

for vs in comb_sort[k]:
    xs_bar.append(vs[0])
    ys_bar.append(vs[1])

x_pos = [i for i, _ in enumerate(xs_bar)]

line_index = xs_bar.index(thresh_results[k][-1][0])

# plt.bar(x_pos, ys_bar, color='green')
# plt.barh(ys_bar,x_pos, color='green')
# plt.xlabel('Function Flow')
# plt.ylabel('Percentage')
# plt.title('Percentage of Function-Flow')

fig, ax = plt.subplots()

ax.barh(x_pos, ys_bar, align='center')
ax.set_yticks(x_pos)
ax.set_yticklabels(xs_bar)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Percentage')
ax.set_title('Percentage of Function Flow in {0}'.format(k))

# horizontal line indicating the threshold
# ax.plot([0., threshold], [threshold, threshold], "k--")

ax.axhline(line_index + 0.5, color="black")
# plt.xticks(x_pos, xs_bar)

plt.show()



# Writing results with stuff from Rob meeting

# titles = ['comp','func-flow','prob','Number of CFF occurrences','Total Items in Dataset','Total CFF counts', 'averaged(/{0})'.format(average),
#           'scaled(/{0})'.format(max_component), '# of unique CFFs','average # CFFs','normalized occurrences',
#           'normalized unique CFFs','# unique CFFs in thresh','ratio unique','harmonic mean']



# write_results_from_dict(combined_thresh_results, 'heating_element_counts8.csv',titles)





# test = pd.DataFrame.from_dict(combined, orient='index', columns=['Total CFF counts', 'averaged(/{0})'.format(average), 'scaled(/{0})'.format(max_component), '# of unique CFFs','average # CFFs'])

# test.to_csv('reservoir_counts.csv', index=True)




# for k,v in thresh_results.items():
#
#     scaled_conf[k][v] = thresh_results[k][v]*(v/max_component)







