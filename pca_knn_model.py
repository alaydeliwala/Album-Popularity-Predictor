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
PCA_NUMBER = 3

# Load the data in
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

train_df = pd.read_csv('data/train_data.csv',
                       usecols=[
                           'acousticness_mean', 'danceability_mean',
                           'energy_mean', 'instrumentalness_mean',
                           'liveness_mean', 'loudness_mean',
                           'speechiness_mean', 'tempo_mean', 'rank'
                       ],
                       nrows=int(len(train_df) / 2 + 1))
test_df = pd.read_csv('data/test_data.csv',
                      usecols=[
                          'acousticness_mean', 'danceability_mean',
                          'energy_mean', 'instrumentalness_mean',
                          'liveness_mean', 'loudness_mean', 'speechiness_mean',
                          'tempo_mean', 'rank'
                      ],
                      nrows=int(len(train_df) / 2 + 1))

train_df_1 = pd.read_csv('data/train_data.csv',
                         usecols=[4, 6, 8, 10, 12, 14, 16, 18, 19],
                         skiprows=int(len(train_df) / 2))
test_df_1 = pd.read_csv('data/test_data.csv',
                        usecols=[4, 6, 8, 10, 12, 14, 16, 18, 19],
                        skiprows=int(len(train_df) / 2))

x_train = np.array(train_df.loc[:, train_df.columns[:-1]])
y_train = np.array(train_df.loc[:, train_df.columns[-1]])

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

# Calculate the covariance matrix of the new training data
for i in sel_pre:
    if pre_dict[i] == 'Normalization':
        # print('Normalization: ')
        cov_norm_train = np.cov(norm_train.T)
        # print('Size of covariance matrix: ', np.shape(cov_norm_train))

    elif pre_dict[i] == 'Standardization':
        # print('Standardization: ')
        cov_z_train = np.cov(z_train.T)
        # print('Size of covariance matrix: ', np.shape(cov_z_train))

for i in sel_pre:
    if pre_dict[i] == 'Normalization':
        # print('Normalization: ')
        norm_eig_vals, norm_eig_vecs = np.linalg.eig(cov_norm_train)
        # print('Eigenvalues: \n', norm_eig_vals[0:5])
        # print('Eigenvectors: \n', norm_eig_vecs[0:5, 0:5], '\n')
        # fig = plt.figure(figsize=(8, 6))
        # plt.plot(np.arange(len(norm_eig_vals)), norm_eig_vals)

    elif pre_dict[i] == 'Standardization':
        # print('Standardization: ')
        z_eig_vals, z_eig_vecs = np.linalg.eig(cov_z_train)
        # print('Eigenvalues: \n', z_eig_vals[0:5])
        # print('Eigenvectors: \n', z_eig_vecs[0:5, 0:5], '\n')
        # fig = plt.figure(figsize=(8, 6))
        # plt.plot(np.arange(len(z_eig_vals)), z_eig_vals)

    # ax = plt.gca()
    # ax.grid(linestyle='--')
    # plt.xlabel('$i^th$ Principle Component', fontsize=16)
    # plt.ylabel('Eigenvalues', fontsize=16)
    # plt.title('Eigenvalues of PCs with ' + pre_dict[i], fontsize=16)
    # plt.savefig('output/elbow_' + pre_dict[i] + '.png')

pc_list = [PCA_NUMBER]
acc_fold_1 = []
for i in sel_pre:
    if pre_dict[i] == 'Normalization':
        print('Normalization FOLD 1: ')
        eig_vecs = norm_eig_vecs
        train_data = norm_train
        test_data = norm_test
    elif pre_dict[i] == 'Standardization':
        print('Standardization FOLD 1: ')
        eig_vecs = z_eig_vecs
        train_data = z_train
        test_data = z_test

    for k in range(len(pc_list)):
        temp_pc_num = pc_list[k]
        pca_train = train_data.dot(eig_vecs[:, 0:temp_pc_num])
        pca_test = test_data.dot(eig_vecs[:, 0:temp_pc_num])

        clf = KNeighborsClassifier(n_neighbors=10)

        clf = clf.fit(pca_train, y_train)
        y_pred = clf.predict(pca_test)

        # Converts album to hit of not based on rank
        for i in range(len(y_pred)):
            if (y_pred[i] > HIT_ALBUM_RANK):
                y_pred[i] = 0
            else:
                y_pred[i] = 1

        for i in range(len(y_test)):
            if (y_test[i] > HIT_ALBUM_RANK):
                y_test[i] = 0
            else:
                y_test[i] = 1
        acc_fold_1.append(float(sum(y_pred == y_test) / len(y_test)))

    # print('Accuracy when PC = 10: ', acc[3])
    # print('Accuracy = {} with {} Principle Components'.format(
    # max(acc_fold_1), pc_list[0]))
    print("Fold 1 Accuracies " + str(acc_fold_1))

#########

x_train = np.array(train_df_1.loc[:, train_df_1.columns[:-1]])
y_train = np.array(train_df_1.loc[:, train_df_1.columns[-1]])

x_test = np.array(test_df_1.loc[:, test_df_1.columns[:-1]])
y_test = np.array(test_df_1.loc[:, test_df_1.columns[-1]])

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

# Calculate the covariance matrix of the new training data
for i in sel_pre:
    if pre_dict[i] == 'Normalization':
        # print('Normalization: ')
        cov_norm_train = np.cov(norm_train.T)
        # print('Size of covariance matrix: ', np.shape(cov_norm_train))

    elif pre_dict[i] == 'Standardization':
        # print('Standardization: ')
        cov_z_train = np.cov(z_train.T)
        # print('Size of covariance matrix: ', np.shape(cov_z_train))

for i in sel_pre:
    if pre_dict[i] == 'Normalization':
        # print('Normalization: ')
        norm_eig_vals, norm_eig_vecs = np.linalg.eig(cov_norm_train)
        # print('Eigenvalues: \n', norm_eig_vals[0:5])
        # print('Eigenvectors: \n', norm_eig_vecs[0:5, 0:5], '\n')
        # fig = plt.figure(figsize=(8, 6))
        # plt.plot(np.arange(len(norm_eig_vals)), norm_eig_vals)

    elif pre_dict[i] == 'Standardization':
        # print('Standardization: ')
        z_eig_vals, z_eig_vecs = np.linalg.eig(cov_z_train)
        # print('Eigenvalues: \n', z_eig_vals[0:5])
        # print('Eigenvectors: \n', z_eig_vecs[0:5, 0:5], '\n')
        # fig = plt.figure(figsize=(8, 6))
        # plt.plot(np.arange(len(z_eig_vals)), z_eig_vals)

    # ax = plt.gca()
    # ax.grid(linestyle='--')
    # plt.xlabel('$i^th$ Principle Component', fontsize=16)
    # plt.ylabel('Eigenvalues', fontsize=16)
    # plt.title('Eigenvalues of PCs with ' + pre_dict[i], fontsize=16)
    # plt.savefig('output/elbow_' + pre_dict[i] + '.png')

pc_list = [PCA_NUMBER]

acc_fold_2 = []
for i in sel_pre:
    if pre_dict[i] == 'Normalization':
        print('Normalization FOLD 2: ')
        eig_vecs = norm_eig_vecs
        train_data = norm_train
        test_data = norm_test
    elif pre_dict[i] == 'Standardization':
        print('Standardization FOLD 2: ')
        eig_vecs = z_eig_vecs
        train_data = z_train
        test_data = z_test

    for k in range(len(pc_list)):
        temp_pc_num = pc_list[k]
        pca_train = train_data.dot(eig_vecs[:, 0:temp_pc_num])
        pca_test = test_data.dot(eig_vecs[:, 0:temp_pc_num])

        clf = KNeighborsClassifier(n_neighbors=10)

        clf = clf.fit(pca_train, y_train)
        y_pred = clf.predict(pca_test)

        # Converts album to hit of not based on rank
        for i in range(len(y_pred)):
            if (y_pred[i] > HIT_ALBUM_RANK):
                y_pred[i] = 0
            else:
                y_pred[i] = 1

        for i in range(len(y_test)):
            if (y_test[i] > HIT_ALBUM_RANK):
                y_test[i] = 0
            else:
                y_test[i] = 1
        acc_fold_2.append(float(sum(y_pred == y_test) / len(y_test)))
    # print("Fold 2 Accuracies " + str(acc_fold_2))

    # print('Accuracy when PC = 10: ', acc[3])
    # print('Accuracy = {} with {} Principle Components'.format(
    # max(acc_fold_2), pc_list[0]))
    print("Fold 2 Accuracies " + str(acc_fold_2))

    for i in sel_pre:
        if pre_dict[i] == 'Normalization':
            print("[NORMALIZATION] 2 FOLD CV Accuracy " +
                  str((acc_fold_2[0] + acc_fold_1[0]) / 2))
        elif pre_dict[i] == 'Standardization':

            print("[STANDARDIZATION] 2 FOLD CV Accuracy " +
                  str((acc_fold_1[1] + acc_fold_2[-1]) / 2))
