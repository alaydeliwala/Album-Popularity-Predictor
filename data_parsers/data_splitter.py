from sklearn.utils import shuffle
import pandas as pd

album_df = pd.read_csv('../data/full_album_data.csv')
# Shuffles the data
shuffled_album_df = shuffle(album_df)

train_len = int(len(shuffled_album_df) * .7)
training_df = shuffled_album_df[:train_len]
testing_df = shuffled_album_df[train_len:]
print(len(shuffled_album_df))
print(len(training_df))
print(len(testing_df))

training_df.to_csv('../data/train_data.csv', index=False)
testing_df.to_csv('../data/test_data.csv', index=False)
