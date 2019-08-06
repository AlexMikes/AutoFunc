
"""

Import csv queries into Pandas Data Frames

"""

import pandas as pd
import os


def get_data(file):

    """
        Takes a .csv file and exports a Pandas data frame and a list of the information

        Parameters
        ----------
        file : string
            A .csv file of a SQL query

        Returns
        -------
        store_data
            Returns a Pandas  data frame of the data in the .csv file
        records
            Returns a list of the data in the .csv file

    """

    # Read in dataset
    store_data = pd.read_csv(os.path.expanduser(file))
    store_data.head()

    # Convert dataset to a list
    records = []

    for i in range(len(store_data)):
        records.append([str(store_data.values[i, j]) for j in range(len(store_data.columns))])



    store_data_no_ids = store_data[['comp', 'func']].copy()

    records_no_ids = []

    for i in range(len(store_data_no_ids)):
        records_no_ids.append([str(store_data_no_ids.values[i, j]) for j in range(len(store_data_no_ids.columns))])



    return store_data, records, store_data_no_ids, records_no_ids


