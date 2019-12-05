import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
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
    
gini_norm = DecisionTreeClassifier(criterion = "gini", random_state = 100) # max_depth, min_samples_leaf
gini_std = DecisionTreeClassifier(criterion = "gini", random_state = 100) # max_depth, min_samples_leaf

gini_norm.fit(norm_train, y_train)
gini_std.fit(z_train, y_train)

entropy_norm = DecisionTreeClassifier(criterion = "entropy", random_state = 100)# max_depth, min_samples_leaf
entropy_std = DecisionTreeClassifier(criterion = "entropy", random_state = 100)# max_depth, min_samples_leaf

entropy_norm.fit(norm_train, y_train)
entropy_std.fit(z_train, y_train)

rank_pred_gini_norm = gini_norm.predict(x_test)
rank_pred_gini_std = gini_std.predict(x_test)

rank_pred_entropy_norm = entropy_norm.predict(x_test)
rank_pred_entropy_std = entropy_std.predict(x_test)

y_pred_gini_norm_hit = [1 if x <= hit_rating else 0 for x in rank_pred_gini_norm]
y_pred_gini_std_hit = [1 if x <= hit_rating else 0 for x in rank_pred_gini_std]

y_pred_entropy_norm_hit = [1 if x <= hit_rating else 0 for x in rank_pred_entropy_norm]
y_pred_entropy_std_hit = [1 if x <= hit_rating else 0 for x in rank_pred_entropy_std]

y_test_hit = [1 if x <= hit_rating else 0 for x in test_df['rank']]

norm_gini_accuracy = accuracy_score(y_test_hit, y_pred_gini_norm_hit) * 100
std_gini_accuracy = accuracy_score(y_test_hit, y_pred_gini_std_hit) * 100
norm_entropy_accuracy = accuracy_score(y_test_hit, y_pred_entropy_norm_hit) * 100
std_entropy_accuracy = accuracy_score(y_test_hit, y_pred_entropy_std_hit) * 100

print('Normalized Gini Accuracy: ', norm_gini_accuracy)
print('Standardized Gini Accuracy: ', std_gini_accuracy)
print('Normalized Entropy Accuracy: ', norm_entropy_accuracy)
print('Standardized Entropy Accuracy: ', std_entropy_accuracy)