#
# For test and train data, creates csv where rank for each song is either 1 (top HIT_SONG song) or 0
#

import pandas as pd

HIT_SONG = 25

train_df = pd.read_csv('../data/train_data.csv')
test_df = pd.read_csv('../data/test_data.csv')

train_df['rank'] = (train_df['rank'] <= HIT_SONG).astype(int)
test_df['rank'] = (test_df['rank'] <= HIT_SONG).astype(int)

train_df.to_csv('../data/train_data_categorical.csv')
test_df.to_csv('../data/test_data_categorical.csv')