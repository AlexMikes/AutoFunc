"""
This reformats a dataframe to a list for certain operations that require that structure

"""


def df_to_list(df):

    """
        Converts the items in a dataframe to items in a list

        Parameters
        ----------
        df : Pandas dataframe
            A Pandas dataframe with the product information

        Returns
        -------
        records_no_ids
            Returns a list with component and function-flow information, without the product id

    """

    store_data_no_ids = df[['comp', 'func']].copy()

    records_no_ids = []

    for i in range(len(store_data_no_ids)):
        records_no_ids.append([str(store_data_no_ids.values[i, j]) for j in range(len(store_data_no_ids.columns))])


    return records_no_ids