
"""

Convert csv file into Pandas Data Frames

"""

import pandas as pd
import os


def make_df(file):

    """
        Takes a .csv file and exports a Pandas data frame

        Parameters
        ----------
        file : string
            A .csv file of a SQL query

        Returns
        -------
        store_data
            Returns a Pandas  data frame of the data in the .csv file

    """

    # Read in dataset
    data_frame = pd.read_csv(os.path.expanduser(file))

    return data_frame


