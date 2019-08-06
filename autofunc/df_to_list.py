
def df_to_list(df):

    store_data_no_ids = df[['comp', 'func']].copy()

    records_no_ids = []

    for i in range(len(store_data_no_ids)):
        records_no_ids.append([str(store_data_no_ids.values[i, j]) for j in range(len(store_data_no_ids.columns))])


    return records_no_ids