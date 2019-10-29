import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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
                      
hit_rating = 25

x_train = np.array(train_df.loc[:, train_df.columns[:-1]])
y_train = np.array(train_df.loc[:, train_df.columns[-1]])

x_test = np.array(test_df.loc[:, test_df.columns[:-1]])
y_test = np.array(test_df.loc[:, test_df.columns[-1]])

pre_dict = {0: 'Normalization', 1: 'Standardization'}
sel_pre = [0, 1]
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

svm_norm = SVC()
svm_std = SVC()

svm_norm.fit(norm_train, y_train)
svm_std.fit(z_train, y_train)

norm_svm_rank_pred = svm_norm.predict(x_test)
std_svm_rank_pred = svm_std.predict(x_test)

y_pred_svm_norm_hit = [1 if x <= hit_rating else 0 for x in norm_svm_rank_pred]
y_pred_svm_std_hit = [1 if x <= hit_rating else 0 for x in std_svm_rank_pred]

y_test_hit = [1 if x <= hit_rating else 0 for x in test_df['rank']]

norm_svm_accuracy = accuracy_score(y_test_hit, y_pred_svm_norm_hit) * 100
std_svm_accuracy = accuracy_score(y_test_hit, y_pred_svm_std_hit) * 100

print('Normalized SVM Accuracy: ', norm_svm_accuracy)
print('Standardized SVM Accuracy: ', std_svm_accuracy)
