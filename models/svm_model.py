import pandas as pd
import numpy as np
import threading
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

HIT_RATING = 25

# For each kernel, store the highest resulting accuracy and its corresponding parameter combination(s) for
# both the standardized data set
max_accuracy_linear_std = 0.0
param_combos_linear_std = []

max_accuracy_poly_std = 0.0
param_combos_poly_std = []

max_accuracy_rbf_std = 0.0
param_combos_rbf_std = []

max_accuracy_sigmoid_std = 0.0
param_combos_sigmoid_std = []

# # # # # # # # # # # # # # # # # #
#                                 #
# Load training and testing data  #
#                                 #
# # # # # # # # # # # # # # # # # #

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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
# For each kernel, tune parameters for each kernel (based on HW3).    #
# Keeps track of the parameter combination(s) for each kernel that    #
# result(s) in the highest accuracy.                                  #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# Performs SVM using the given kernel
def perform_svm(kernel):
    # Parameters we will use to build SVM models
    reg_params = [0.1, 0.2, 0.3, 1, 5, 10, 20, 100, 200, 1000]
    degree_params = [1, 2, 3, 4, 5]
    # coef0_params = [0.0001, 0.001, 0.002, 0.01, 0.02, 0.1, 0.2, 0.3, 1, 2, 5, 10]
    coef0_params = [0.0001, 0.001, 0.01, 0.1, 0.2, 1, 2, 5, 10]
    gamma_params = [0.0001, 0.001, 0.002, 0.01, 0.02, 0.1, 0.2, 1, 2, 3]

    # We store the highest accuracies and the param combo(s) that led to that number as
    # global vars, so declare as such
    global param_combos_linear_std
    global param_combos_poly_std
    global param_combos_rbf_std
    global param_combos_sigmoid_std
    global max_accuracy_linear_std
    global max_accuracy_poly_std
    global max_accuracy_rbf_std
    global max_accuracy_sigmoid_std

    print("Starting ", kernel)

    if kernel == 'linear':
        for reg in reg_params:
            # We will train 2 models; one for normalized data, and one for standardized data
            # svm_norm = SVC(kernel=kernel, gamma='scale', C=reg)
            svm_std = SVC(kernel=kernel, gamma='scale', C=reg)

            # svm_norm.fit(norm_train, y_train)
            svm_std.fit(z_train, y_train)

            # norm_svm_rank_pred = svm_norm.predict(x_test)
            std_svm_rank_pred = svm_std.predict(z_test)

            # Transform predicted data to 1 if hit song, 0 if not
            # y_pred_svm_norm_hit = [1 if x <= HIT_RATING else 0 for x in norm_svm_rank_pred]
            y_pred_svm_std_hit = [
                1 if x <= HIT_RATING else 0 for x in std_svm_rank_pred]

            # Transform test data to 1 if hit song, 0 if not
            y_test_hit = [1 if x <= HIT_RATING else 0 for x in test_df['rank']]

            # norm_svm_accuracy = accuracy_score(y_test_hit, y_pred_svm_norm_hit) * 100
            std_svm_accuracy = accuracy_score(y_test_hit, y_pred_svm_std_hit)

            # Keep track of our max accuracy for for both datasets
            param_combo = {'kernel': kernel, 'C': reg}

            if std_svm_accuracy == max_accuracy_linear_std:
                param_combos_linear_std.append(param_combo)
            elif std_svm_accuracy > max_accuracy_linear_std:
                param_combos_linear_std = [param_combo]
                max_accuracy_linear_std = std_svm_accuracy

    elif kernel == 'poly':
        for reg in reg_params:
            for degree in degree_params:
                for coef0 in coef0_params:
                    # We will train 2 models; one for normalized data, and one for standardized data
                    # svm_norm = SVC(kernel=kernel, gamma='scale', C=reg, degree=degree, coef0=coef0)
                    svm_std = SVC(kernel=kernel, gamma='scale',
                                  C=reg, degree=degree, coef0=coef0)

                    # svm_norm.fit(norm_train, y_train)
                    svm_std.fit(z_train, y_train)

                    # norm_svm_rank_pred = svm_norm.predict(x_test)
                    std_svm_rank_pred = svm_std.predict(z_test)

                    # Transform predicted data to 1 if hit song, 0 if not
                    # y_pred_svm_norm_hit = [1 if x <= HIT_RATING else 0 for x in norm_svm_rank_pred]
                    y_pred_svm_std_hit = [
                        1 if x <= HIT_RATING else 0 for x in std_svm_rank_pred]

                    # Transform test data to 1 if hit song, 0 if not
                    y_test_hit = [
                        1 if x <= HIT_RATING else 0 for x in test_df['rank']]

                    # norm_svm_accuracy = accuracy_score(y_test_hit, y_pred_svm_norm_hit) * 100
                    std_svm_accuracy = accuracy_score(
                        y_test_hit, y_pred_svm_std_hit)

                    # Keep track of our max accuracy for for both datasets
                    param_combo = {'kernel': kernel, 'C': reg,
                                   'degree': degree, 'coef0': coef0}

                    if std_svm_accuracy == max_accuracy_poly_std:
                        param_combos_poly_std.append(param_combo)
                    elif std_svm_accuracy > max_accuracy_poly_std:
                        param_combos_poly_std = [param_combo]
                        max_accuracy_poly_std = std_svm_accuracy

    elif kernel == 'rbf':
        for reg in reg_params:
            for g in gamma_params:
                # We will train 2 models; one for normalized data, and one for standardized data
                # svm_norm = SVC(kernel=kernel, C=reg, gamma=g)
                svm_std = SVC(kernel=kernel, C=reg, gamma=g)

                # svm_norm.fit(norm_train, y_train)
                svm_std.fit(z_train, y_train)

                # norm_svm_rank_pred = svm_norm.predict(x_test)
                std_svm_rank_pred = svm_std.predict(z_test)

                # Transform predicted data to 1 if hit song, 0 if not
                # y_pred_svm_norm_hit = [1 if x <= HIT_RATING else 0 for x in norm_svm_rank_pred]
                y_pred_svm_std_hit = [
                    1 if x <= HIT_RATING else 0 for x in std_svm_rank_pred]

                # Transform test data to 1 if hit song, 0 if not
                y_test_hit = [
                    1 if x <= HIT_RATING else 0 for x in test_df['rank']]

                # norm_svm_accuracy = accuracy_score(y_test_hit, y_pred_svm_norm_hit) * 100
                std_svm_accuracy = accuracy_score(
                    y_test_hit, y_pred_svm_std_hit)

                # Keep track of our max accuracy for for both datasets
                param_combo = {'kernel': kernel, 'C': reg, 'gamma': g}

                if std_svm_accuracy == max_accuracy_rbf_std:
                    param_combos_rbf_std.append(param_combo)
                elif std_svm_accuracy > max_accuracy_rbf_std:
                    param_combos_rbf_std = [param_combo]
                    max_accuracy_rbf_std = std_svm_accuracy

    elif kernel == 'sigmoid':
        for reg in reg_params:
            for coef0 in coef0_params:
                for g in gamma_params:
                    # We will train 2 models; one for normalized data, and one for standardized data
                    # svm_norm = SVC(kernel=kernel, C=reg, coef0=coef0, gamma=g)
                    svm_std = SVC(kernel=kernel, C=reg, coef0=coef0, gamma=g)

                    # svm_norm.fit(norm_train, y_train)
                    svm_std.fit(z_train, y_train)

                    # norm_svm_rank_pred = svm_norm.predict(x_test)
                    std_svm_rank_pred = svm_std.predict(z_test)

                    # Transform predicted data to 1 if hit song, 0 if not
                    # y_pred_svm_norm_hit = [1 if x <= HIT_RATING else 0 for x in norm_svm_rank_pred]
                    y_pred_svm_std_hit = [
                        1 if x <= HIT_RATING else 0 for x in std_svm_rank_pred]

                    # Transform test data to 1 if hit song, 0 if not
                    y_test_hit = [
                        1 if x <= HIT_RATING else 0 for x in test_df['rank']]

                    # norm_svm_accuracy = accuracy_score(y_test_hit, y_pred_svm_norm_hit) * 100
                    std_svm_accuracy = accuracy_score(
                        y_test_hit, y_pred_svm_std_hit)

                    # Keep track of our max accuracy for for both datasets
                    param_combo = {'kernel': kernel,
                                   'C': reg, 'coef0': coef0, 'gamma': g}

                    if std_svm_accuracy == max_accuracy_sigmoid_std:
                        param_combos_sigmoid_std.append(param_combo)
                    elif std_svm_accuracy > max_accuracy_sigmoid_std:
                        param_combos_sigmoid_std = [param_combo]
                        max_accuracy_sigmoid_std = std_svm_accuracy

    print("Ending ", kernel)


# # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                     #
# Spawn a separate thread for each of the 4 kernels,  #
# and run SVM.                                        #
#                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # #

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
threads = []
for i in range(4):
    t = threading.Thread(target=perform_svm, args=(kernels[i],))
    threads.append(t)
    t.start()

for thread in threads:
    thread.join()

print("Linear Kernel: ")
print(max_accuracy_linear_std)
print(param_combos_linear_std)

print("Poly kernel: ")
print(max_accuracy_poly_std)
print(param_combos_poly_std)

print("Rbf kernel: ")
print(max_accuracy_rbf_std)
print(param_combos_rbf_std)

print("Sigmoid kernel: ")
print(max_accuracy_sigmoid_std)
print(param_combos_sigmoid_std)

# # # # # # # # #
#               #
# Graph results #
#               #
# # # # # # # # #
all_accuracy_values = [max_accuracy_linear_std, max_accuracy_poly_std,
                       max_accuracy_rbf_std, max_accuracy_sigmoid_std]
plt.bar(kernels, all_accuracy_values)
plt.xlabel("Accuracy")
plt.ylabel("Kernel")
plt.title("SVM Accuracy w/ Standardized Data")
plt.savefig('output/SVM_Accuracy_with_Standardized_data.png')
