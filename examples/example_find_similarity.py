from autofunc.find_similarities import find_similarities
import pandas as pd
import os.path

script_dir = os.path.dirname(__file__)
file_to_learn = os.path.join(script_dir, '../assets/consumer_systems.csv')

train_data = pd.read_csv(file_to_learn)

## Make similarity dataframe
similarity_df = find_similarities(train_data)

## This can take a while but never changes for each dataset, so uncomment this line to save to a csv
similarity_df.to_csv('consumer_similarity1.csv', index = True, index_label=False, header= True)