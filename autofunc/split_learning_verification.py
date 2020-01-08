"""

Split a dataframe into training and testing sets

"""

def split_learning_verification(dataframe, verification_ids):

    """
        Takes a Pandas dataframe and splits it into training and testing sets. The input IDs are the testing set, the
        rest of the dataframe is the training set

        Parameters
        ----------
        dataframe : Pandas dataframe
            A Pandas dataframe of the whole set that will be split

        verification_ids : list
            The ID(s) that will be separated from the dataframe to constitute the testing set

        Returns
        -------
        test_df
            Returns a Pandas  dataframe with the testing set consisting of the products with the IDs in the input list

        train_df
            Returns a Pandas dataframe with the training set consisting of every product with IDs that were not
            in the input list

    """

    ids = list(map(int, dataframe.id.unique()))

    learn_ids = []

    for e in ids:
        if e not in verification_ids:
            learn_ids.append(e)


    test_df = dataframe[dataframe['id'].isin(verification_ids)]


    train_df = dataframe[dataframe['id'].isin(learn_ids)]

    return test_df, train_df