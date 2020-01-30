# This model first standardizes the data and then
# then uses K-NN with K ranging from
# determine if an album will be a hit or not
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('agg')

HIT_ALBUM_RANK = 25

# Load the data into pandas DataFrames
train_df = pd.read_csv('../data/train_data.csv',
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

print('Training sample size: ', len(train_df))
print('Testing sample size: ', len(test_df))

# Divide training data and labels
x_train = np.array(train_df.loc[:, train_df.columns[:-1]])
y_train = np.array(train_df.loc[:, train_df.columns[-1]])

# Divide testing data and labels
x_test = np.array(test_df.loc[:, test_df.columns[:-1]])
y_test = np.array(test_df.loc[:, test_df.columns[-1]])

# z_train is the standardized training data
z_train = stats.zscore(x_train)
# Use the min and max of training data to standardize testing data
mean_train = np.mean(x_train, axis=0)
std_train = np.std(x_train, axis=0)
# z_test is the standardized testing data
z_test = (x_test - mean_train) / std_train

# Transforms the labels based on Hit Album Rank Criteria
y_train = [1 if x <= HIT_ALBUM_RANK else 0 for x in y_train]
y_test = [1 if x <= HIT_ALBUM_RANK else 0 for x in y_test]

# A dictionary of k values and accuracies
acc = {}
cv_acc = {}
prec = {}

# Iterates through the diffrent K values = [1,16]
for neighbors in range(1, 17):
    clf = KNeighborsClassifier(n_neighbors=(neighbors))

    cv_score = cross_val_score(
        clf, z_train, y_train, cv=4, scoring='accuracy').mean()
    cv_acc[str(neighbors)] = cv_score
    plt.plot((neighbors), (cv_acc[str(neighbors)]), 'bo')

    # Calculates the testing accuracy
    clf = clf.fit(z_train, y_train)
    y_pred = clf.predict(z_test)
    acc[str(neighbors)] = float(sum(y_pred == y_test) / len(y_test))
    prec[str(neighbors)] = precision_score(y_test, y_pred)
    plt.plot((neighbors), (float(sum(y_pred == y_test) / len(y_test))), 'ro')

    # Used to see how many times the model predicts 0
    # num_0 = 0
    # for i in y_pred:
    #     if i == 0:
    #         num_0+=1
    # print(str(neighbors) + "-NN has predicted " + str(num_0) +" entries as 0 (not a hit)")

plt.title("K-NN Accuracy w/ Standardized Data and 10-Fold CV")
plt.axis([0, 17, .7, 1])
ax = plt.gca()
ax.set_autoscale_on(False)
plt.ylabel("Accuracy")
plt.xlabel("Value of K")
plt.legend(['Training Accuracy', 'Testing Accuracy'], loc='lower right')
plt.savefig('output/KNN_Accuracy_with_Standardized_data.png')


v = list(acc.values())
k = list(acc.keys())
print("The value of K that provides the highest accuracy is " +
      k[v.index(max(v))] + " with an accuracy of " +
      str(round(acc[str(k[v.index(max(v))])] * 100, 5)) + "%")

v = list(prec.values())
k = list(prec.keys())
print("The value of K that provides the highest precision is " +
      k[v.index(max(v))] + " with an accuracy of " +
      str(round(prec[str(k[v.index(max(v))])] * 100, 5)) + "%")

print()
print("k value | train acc | test acc  | test precision")
for test_acc in acc:
    print("{0}       | {1}    | {2}  | {3}".format(test_acc, round(
        cv_acc[test_acc], 5), round(acc[test_acc], 5), round(prec[test_acc], 5)))
