"""
Builds a similarity matrix with product IDs as the rows and headers and the similarity between each combination as
the matrix value in that index. The diagonal is 1 because each product is 100% similar to itself.

Similarity here is defined as the percentage of components that two products have in common. The matrix is not symmetric
because each product can have a different number of components.

"""


import pandas as pd
from numpy import zeros

def find_similarities(input_dataframe):


    """
        Find the similarity between all products in a repository

        Parameters
        ----------
        input_dataframe : Pandas dataframe
            A Pandas dataframe with the product information

        Returns
        -------
        similarity_df
            Returns a Pandas dataframe in an nxn matrix format with the similarity between each product

    """

    # Separating all of the product ids into a list
    all_ids = list(map(int, input_dataframe.id.unique()))

    # Building an nxn dataframe with the product ids as the row and column headers
    similarity_df = pd.DataFrame(zeros((len(all_ids), len(all_ids))), columns=all_ids,
                                 index=all_ids)


    for e in all_ids:

        testing_comps = input_dataframe.loc[input_dataframe['id'] == e]['comp']
        unq_testing_comps = list(testing_comps.unique())

        for id in all_ids:

            comparing_comps = input_dataframe.loc[input_dataframe['id'] == id]['comp']
            unq_comparing_comps = list(comparing_comps.unique())

            similar = 0

            for tc in unq_testing_comps:
                if tc in unq_comparing_comps:
                    similar += 1

            # Count components in each, find the percent of similar components
            # Flipping rows and columns to use Pandas built in evaluators for thresholding
            percent_similar = similar/len(unq_comparing_comps)

            # Put the similarity value into the matrix at the correct place
            similarity_df.loc[e,id]=percent_similar

            print(percent_similar)

    return similarity_df