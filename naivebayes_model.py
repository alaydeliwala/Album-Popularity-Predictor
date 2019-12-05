import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from scipy import stats

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

# # # # # # # # # # #
#                   #
# Standardize data  #
#                   #
# # # # # # # # # # #

z_train = stats.zscore(x_train)
# Use the min and max of training data to standardize testing data
mean_train = np.mean(x_train, axis=0)
std_train = np.std(x_train, axis=0)
z_test = (x_test - mean_train) / std_train

# # # # # # # # #
#               #
# Apply model   #
#               #
# # # # # # # # #

gnb = GaussianNB()
gnb.fit(z_train, y_train)
y_prediction = gnb.predict(z_test)

y_prediction_hit =[1 if x <= HIT_SONG else 0 for x in y_prediction]
y_test_hit = [1 if x <= HIT_SONG else 0 for x in test_df['rank']]

accuracy = accuracy_score(y_test_hit, y_prediction_hit) * 100

print("Gaussian Naive-bayes Accuracy: ", accuracy)

