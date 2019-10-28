import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

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

gini = DecisionTreeClassifier(criterion = "gini", random_state = 100) # max_depth, min_samples_leaf
gini.fit(x_train, y_train)

entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100)# max_depth, min_samples_leaf
entropy.fit(x_train, y_train)

rank_pred_gini = gini.predict(x_test)
rank_pred_entropy = entropy.predict(x_test)

y_pred_gini_hit = [1 if x <= hit_rating else 0 for x in rank_pred_gini]

y_pred_entropy_hit =[1 if x <= hit_rating else 0 for x in rank_pred_entropy]

y_test_hit = [1 if x <= hit_rating else 0 for x in test_df['rank']]

gini_accuracy = accuracy_score(y_test_hit, y_pred_gini_hit) * 100
entropy_accuracy = accuracy_score(y_test_hit, y_pred_entropy_hit) * 100

print('Gini Accuracy: ', gini_accuracy)
print('Entropy Accuracy: ', entropy_accuracy)
