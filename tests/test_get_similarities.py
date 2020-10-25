from autofunc.find_similarities import find_similarities
import pandas as pd
import os.path


def test1():

    """ Testing to ensure the main diagonal of the similarity matrix is 1"""

    script_dir = os.path.dirname(__file__)
    file_to_learn = os.path.join(script_dir, '../autofunc/assets/blade_systems.csv')

    train_data = pd.read_csv(file_to_learn)

    ## Make similarity dataframe
    similarity_df = find_similarities(train_data)

    assert similarity_df.iat[0,0] == 1.0

