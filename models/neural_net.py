# first neural network with keras make predictions
from keras.models import Sequential
from keras.layers import Dense
from scipy import stats
import pandas as pd
import numpy as np
from statistics import mean
from sklearn.metrics import accuracy_score
import os


# change ranks
df = pd.read_csv('../data/train_data.csv')

df.loc[df['rank'] <= 25, 'rank'] = 1
df.loc[df['rank'] > 25, 'rank'] = 0
df.to_csv('../data/tmp_train_data.csv')


train_df = pd.read_csv('../data/tmp_train_data.csv',
                       usecols=[
                           'acousticness_mean', 'danceability_mean',
                           'energy_mean', 'instrumentalness_mean',
                           'liveness_mean', 'loudness_mean',
                           'speechiness_mean', 'tempo_mean', 'rank'
                       ])
test_df = pd.read_csv('../data/test_data.csv',
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
z_train = stats.zscore(x_train)

# Use the min and max of training data to standardize testing data
mean_train = np.mean(x_train, axis=0)
std_train = np.std(x_train, axis=0)
z_test = (x_test - mean_train) / std_train

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
#model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# compile the keras model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(z_train, y_train, epochs=10, batch_size=10)


predictions = model.predict(z_test)
print
y_test_hit = [1 if x <= 25 else 0 for x in y_test]
y_predictions = [1 if x <= 25 else 0 for x in predictions]
accuracy = accuracy_score(y_test_hit, y_predictions) * 100
print('Deep Neural Network Accuracy: ', accuracy)

for i in range(30):
    print('%s => %d (expected %d)' %
          (x_train[i].tolist(), predictions[i], y_train[i]))

# remove the temp train data
os.system("rm ../data/tmp_train_data.csv")
