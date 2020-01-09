
"""

Convert csv file into Pandas Data Frames. Removes some of the erroneous data from The Design Repository queries

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
    df = pd.read_csv(os.path.expanduser(file))

    df = df[df.comp != 'unclassified']
    df = df[df.comp != 'system']
    df = df[df.comp != 'assembly']
    df = df[~df.comp.str.contains('output')]


    return df


