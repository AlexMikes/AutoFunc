

def split_learning_verification(dataframe, verification_ids):



    ids = list(map(int, dataframe.id.unique()))

    learn_ids = []

    # for e in verification_ids:
    #     if e not in ids:
    #         # continue
    #         print('Skipping {0:.2f}'.format(e))
    #         # continue
    #         # raise ValueError('The verification ids are not in the learning set')

    for e in ids:
        if e not in verification_ids:
            learn_ids.append(e)


    test_df = dataframe[dataframe['id'].isin(verification_ids)]


    train_df = dataframe[dataframe['id'].isin(learn_ids)]

    return test_df, train_df