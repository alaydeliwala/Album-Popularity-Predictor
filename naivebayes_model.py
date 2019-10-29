import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
# from data_parsers import data_splitter_categorical as split # Import our cut-off number for a hit song

HIT_SONG = 25

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

x_train = np.array(train_df.loc[:, train_df.columns[:-1]])
y_train = np.array(train_df.loc[:, train_df.columns[-1]])

x_test = np.array(test_df.loc[:, test_df.columns[:-1]])
y_test = np.array(test_df.loc[:, test_df.columns[-1]])

gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_prediction = gnb.predict(x_test)

y_prediction_hit =[1 if x <= HIT_SONG else 0 for x in y_prediction]
y_test_hit = [1 if x <= HIT_SONG else 0 for x in test_df['rank']]

accuracy = accuracy_score(y_test_hit, y_prediction_hit) * 100

print("Gaussian Naive-bayes Accuracy: ", accuracy)

#
#
# Runs Naive-Bayes when song rank is either 1 or 0 in train/test data... predicts all 0s though
#
#

train_df = pd.read_csv('data/train_data_categorical.csv',
                       usecols=[
                           'acousticness_mean', 'danceability_mean',
                           'energy_mean', 'instrumentalness_mean',
                           'liveness_mean', 'loudness_mean',
                           'speechiness_mean', 'tempo_mean', 'rank'
                       ])
test_df = pd.read_csv('data/test_data_categorical.csv',
                      usecols=[
                          'acousticness_mean', 'danceability_mean',
                          'energy_mean', 'instrumentalness_mean',
                          'liveness_mean', 'loudness_mean', 'speechiness_mean',
                          'tempo_mean', 'rank'
                      ])

x_train = np.array(train_df.loc[:, train_df.columns[:-1]])
y_train = np.array(train_df.loc[:, train_df.columns[-1]])

x_test = np.array(test_df.loc[:, test_df.columns[:-1]])
y_test = np.array(test_df.loc[:, test_df.columns[-1]])

bnb = BernoulliNB()
bnb.fit(x_train, y_train)
y_prediction = bnb.predict(x_test)

accuracy = accuracy_score(y_test, y_prediction) * 100

print("Bernoulli Naive-bayes Accuracy: ", accuracy)

# print("Bernoulli Naive-bayes Report: ")
# print(classification_report(y_test, y_prediction))

