import pandas as pd
from numpy import zeros

def find_similarities(input_dataframe):


    all_ids = list(map(int, input_dataframe.id.unique()))

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

            # Flipping rows and columns to use Pandas built in evaluators for thresholding
            percent_similar = similar/len(unq_comparing_comps)

            similarity_df.loc[e,id]=percent_similar

    return similarity_df