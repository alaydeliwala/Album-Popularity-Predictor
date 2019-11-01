import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

HIT_ALBUM_RANK = 50

# Load the data into pandas DataFrames
train_df = pd.read_csv('data/train_data.csv',
                       usecols=[
                           'acousticness_mean', 'danceability_mean',
                           'energy_mean', 'instrumentalness_mean',
                           'liveness_mean', 'loudness_mean',
                           'speechiness_mean', 'tempo_mean', 'rank'
                       ])
test_df = pd.read_csv('data/test_data.csv',
                      usecols=[
                          'acousticness_mean', 'danceability_mean',
                          'energy_mean', 'instrumentalness_mean',
                          'liveness_mean', 'loudness_mean', 'speechiness_mean',
                          'tempo_mean', 'rank'
                      ])

print('Training sample size: ', len(train_df))
print('Testing sample size: ', len(test_df))

hit = 0
for rank in train_df['rank']:
    if rank <= HIT_ALBUM_RANK:
        hit += 1
print("Percentage Hits in Training Data " + str(hit / len(train_df) * 100) +
      '%')

for rank in test_df['rank']:
    if rank <= HIT_ALBUM_RANK:
        hit += 1
print("Percentage Hits in Testing Data " + str(hit / len(train_df) * 100) +
      '%')
