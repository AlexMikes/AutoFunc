from autofunc.find_similarities import find_similarities
import pandas as pd
import os.path

script_dir = os.path.dirname(__file__)
file_to_learn = os.path.join(script_dir, '../assets/consumer_systems.csv')

train_data = pd.read_csv(file_to_learn)

## Make similarity dataframe, takes a while so it is saved to csv first for reading in later
similarity_df = find_similarities(train_data)
# similarity_df.to_csv('consumer_similarity.csv', index = True, index_label=False, header= True)