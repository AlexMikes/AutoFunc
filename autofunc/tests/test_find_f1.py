from autofunc.get_precision_recall import precision_recall
from autofunc.get_top_results import get_top_results
from autofunc.counter_pandas import counter_pandas
from autofunc.split_learning_verification import split_learning_verification
from autofunc.df_to_list import df_to_list
from autofunc.make_df import make_df
import pandas as pd
import os.path

def test1():

    """ Test to find F1 score by manually selecting product id(s) from original data to test """

    script_dir = os.path.dirname(__file__)
    file_to_learn = os.path.join(script_dir, '../assets/consumer_systems.csv')

    train_data = make_df(file_to_learn)

    # Use a threshold to get the top XX% of frequency values
    threshold = 0.7

    ## Choose ID(s) from learning file to separate into the testing set
    test_ids  = [691, 169]

    test_df, train_df = split_learning_verification(train_data, test_ids)

    test_list = df_to_list(test_df)

    comb_sort = counter_pandas(train_df)
    thresh_results = get_top_results(comb_sort, threshold)

    # Find the F1 score
    learned_dict, matched, overmatched, unmatched, recall, precision, f1 = precision_recall(thresh_results, test_list)

    assert len(learned_dict) !=0
    assert f1>0




