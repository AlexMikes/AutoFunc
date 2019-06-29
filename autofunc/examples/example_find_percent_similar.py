from autofunc.get_percent_similar import *
import os.path


# Dataset used for data mining
script_dir = os.path.dirname(__file__)
file1 = os.path.join(script_dir, '../assets/bladeCombined_id.csv')
file2 = os.path.join(script_dir, '../assets/jigsawQuery_headers.csv')

test_data = pd.read_csv(file1)

# ids = list(store_data.id.unique())
ids = list(map(int,test_data.id.unique()))

test_id = ids[3]

# learning_comps = test_data.loc[test_data['id'] == test_id]['comp']
#
# unq_learning_comps = list(learning_comps.unique())

ps_thresh = 0.6

keep_ids = []

for id in ids:

    ps = percent_similar(file1,file2, id)

    print(ps)

    if ps > ps_thresh:

        keep_ids.append(id)






