import pandas as pd
import os.path




def unique(filename,col):



    find = store_data[col]

    # ids = list(store_data.id.unique())
    uniques = list(map(int, store_data.id.unique()))

    return uniques



def percent_similar(learning_file, input_file, test_id):

    # # Dataset used for data mining
    # script_dir = os.path.dirname(__file__)
    # file = os.path.join(script_dir, filename)

    learning_set = pd.read_csv(learning_file)

    input_set = pd.read_csv(input_file)

    input_comps = list(input_set.comp.unique())

    # Find rows with product id
    learning_comps = learning_set.loc[learning_set['id'] == test_id]['comp']

    # Find unique components from rows with product id
    unq_learning_comps = list(learning_comps.unique())

    similar = 0

    for e in input_comps:
        if e in unq_learning_comps:
            similar += 1

    percent_similar = similar/len(unq_learning_comps)

    return percent_similar

    # input_counts = {}

    # with open(input_comps, encoding='utf-8-sig') as input_file:
    #     for row in csv.reader(input_file, delimiter=','):
    #
    #         input_comp = row[0]
    #         # Create a dictionary with each component
    #         if comp not in counts:
    #             input_counts[input_comp] = 1
    #         else:
    #             input_counts[input_comp] += 1
    #
    # learning_counts = {}
    #
    # with open(learning_file, encoding='utf-8-sig') as learning:
    #     for row in csv.reader(learning, delimiter=','):
    #
    #         learning_comp = row[0]
    #         # Create a dictionary with each component
    #         if learning_comp not in learning_counts:
    #             learning_counts[learning_comp] = 1
    #         else:
    #             learning_counts[learning_comp] += 1
    #
    # seen = {}
    # similar = 0
    #
    # for k,v in input_counts:
    #
    #     if k in learning_counts:
    #
    #         if k not in seen:
    #
    #             seen[k] += 1
    #
    #             similar += 1






