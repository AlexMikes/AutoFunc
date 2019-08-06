

def split_learning_verification(dataframe, verification_ids):

    ids = list(map(int, dataframe.id.unique()))

    learn_ids = []

    for e in ids:
        if e not in verification_ids:
            learn_ids.append(e)


    ver_df = dataframe[dataframe['id'].isin(verification_ids)]


    learn_df = dataframe[dataframe['id'].isin(learn_ids)]

    return  ver_df, learn_df