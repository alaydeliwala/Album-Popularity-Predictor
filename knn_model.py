# This model first standardizes and normalizes the data and then
# uses PCA to select the top XX features and then uses XX-NN to
# determine if an album will be a hit or not

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

HIT_ALBUM_RANK = 25

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

# Divide training data and labels
x_train = np.array(train_df.loc[:, train_df.columns[:-1]])
y_train = np.array(train_df.loc[:, train_df.columns[-1]])

# Divide testing data and labels
x_test = np.array(test_df.loc[:, test_df.columns[:-1]])
y_test = np.array(test_df.loc[:, test_df.columns[-1]])

pre_dict = {0: 'Normalization', 1: 'Standardization'}
sel_pre = [0, 1]

# Normalizing and Standardizing the data
for i in sel_pre:
    if pre_dict[i] == 'Normalization':
        norm_train = (x_train - x_train.min(axis=0)) / (x_train.max(axis=0) -
                                                        x_train.min(axis=0))
        # Use the min and max of training data to normalize testing data
        min_train = np.min(x_train, axis=0)
        max_train = np.max(x_train, axis=0)
        norm_test = (x_test - min_train) / (max_train - min_train)
    elif pre_dict[i] == 'Standardization':
        z_train = stats.zscore(x_train)
        # Use the min and max of training data to standardize testing data
        mean_train = np.mean(x_train, axis=0)
        std_train = np.std(x_train, axis=0)
        z_test = (x_test - mean_train) / std_train

for i in sel_pre:
    acc = {}

    for neighbors in range(16):

        clf = KNeighborsClassifier(n_neighbors=(neighbors + 1))

        clf = clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        for j in range(len(y_pred)):
            if (y_pred[j] > HIT_ALBUM_RANK):
                y_pred[j] = 0
            else:
                y_pred[j] = 1

        for j in range(len(y_test)):
            if (y_test[j] > HIT_ALBUM_RANK):
                y_test[j] = 0
            else:
                y_test[j] = 1
        acc[str(neighbors + 1)] = float(sum(y_pred == y_test) / len(y_test))
        plt.plot((neighbors + 1), (float(sum(y_pred == y_test) / len(y_test))),
                 'ro')

    if i == 0:
        plt.title("K-NN Accuracy w/ Normalized Data")
        plt.savefig('output/KNN_Accuracy_with_Normalized_data.png')

    elif i == 1:
        plt.title("K-NN Accuracy w/ Standardized Data")
        plt.savefig('output/KNN_Accuracy_with_Standardized_data.png')

    plt.clf()

    # print('Accuracy = {}'.format(max(acc)))