import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

# Reading in the datasets from the data file
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
                
# Global Variable for hit_rating his is the threshold for a hit album
hit_rating = 25

# Split x and y from the training data set
x_train = np.array(train_df.loc[:, train_df.columns[:-1]])
y_train = np.array(train_df.loc[:, train_df.columns[-1]])

# Split x and y from the test data set
x_test = np.array(test_df.loc[:, test_df.columns[:-1]])
y_test = np.array(test_df.loc[:, test_df.columns[-1]])

# Standardize the training data set
z_train = stats.zscore(x_train)
# Use the min and max of training data to standardize testing data
mean_train = np.mean(x_train, axis=0)
std_train = np.std(x_train, axis=0)
z_test = (x_test - mean_train) / std_train

# Create the Gini Index Tree Classifier
gini_std = DecisionTreeClassifier(criterion = "gini", random_state = 100) # max_depth, min_samples_leaf

# Fit the gini index model to the standardized data set
gini_std.fit(z_train, y_train)

# Create the Entropy Tree Classifier
entropy_std = DecisionTreeClassifier(criterion = "entropy", random_state = 100)# max_depth, min_samples_leaf

# Fit the entropy model to the standardized data set
entropy_std.fit(z_train, y_train)

# Make the ranked predictions for the gini model
rank_pred_gini_std = gini_std.predict(z_test)

# Make the ranked predictions for the entropy model
rank_pred_entropy_std = entropy_std.predict(z_test)

# Reclassify predictions and test ranks to hits or not hits
y_pred_gini_std_hit = [1 if x <= hit_rating else 0 for x in rank_pred_gini_std]

y_pred_entropy_std_hit = [1 if x <= hit_rating else 0 for x in rank_pred_entropy_std]

y_test_hit = [1 if x <= hit_rating else 0 for x in test_df['rank']]

# Calculate the accuracy scores
std_gini_accuracy = accuracy_score(y_test_hit, y_pred_gini_std_hit) * 100
std_entropy_accuracy = accuracy_score(y_test_hit, y_pred_entropy_std_hit) * 100

# Print the accuracies for the Gini and Entropy decision trees
print('Standardized Gini Accuracy: ', std_gini_accuracy)
print('Standardized Entropy Accuracy: ', std_entropy_accuracy)

#Create the Gini Index Accuracy Graph
for datapoints in range(2, 16):
    gini_std = DecisionTreeClassifier(criterion = "gini", random_state = 100,
        min_samples_split = datapoints)
        
    gini_std.fit(z_train, y_train)
    
    rank_pred_gini_std = gini_std.predict(z_test)
    
    y_test_hit = [1 if x <= hit_rating else 0 for x in test_df['rank']]
    y_pred_gini_std_hit = [1 if x <= hit_rating else 0 for x in rank_pred_gini_std]
    sum = 0
    for point in range(len(y_test_hit)):
        if(y_pred_gini_std_hit[point] == y_test_hit[point]):
            sum += 1
    avg = sum/len(y_test_hit)
    plt.plot((datapoints), avg,
             'ro')
             
plt.title("GINI Index Accuracy w/ Standardized Data")
plt.xlabel('Datapoints Needed to Split')
plt.ylabel('Accuracy')
plt.savefig('output/GINI_Accuracy_with_Standardized_Data.png')
plt.clf()

#Create the Entropy Accuracy Graph
for datapoints in range(2, 16):
    entropy_std = DecisionTreeClassifier(criterion = "entropy", random_state = 100, min_samples_split = datapoints)
    
    entropy_std.fit(z_train, y_train)

    rank_pred_entropy_std = entropy_std.predict(z_test)

    y_test_hit = [1 if x <= hit_rating else 0 for x in test_df['rank']]

    y_pred_entropy_std_hit = [1 if x <= hit_rating else 0 for x in rank_pred_entropy_std]
    sum = 0
    for point in range(len(y_test_hit)):
        if(y_pred_entropy_std_hit[point] == y_test_hit[point]):
            sum += 1
    avg = sum/len(y_test_hit)
    plt.plot((datapoints), avg,
             'ro')
    
plt.title("Entropy Accuracy W/ Standardized Data")
plt.xlabel('Datapoints Needed to Split')
plt.ylabel('Accuracy')
plt.savefig('output/Entropy_Accuracy_with_Standardized_Data.png')
plt.clf()




    

