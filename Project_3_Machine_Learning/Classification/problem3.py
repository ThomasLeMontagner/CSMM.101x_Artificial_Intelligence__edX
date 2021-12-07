import csv
import sys
from statistics import mean

import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn import svm, linear_model, neighbors, tree, ensemble
from sklearn.utils import shuffle


# def plot_data(X0, X1):
#     plt.scatter(X0[:,0], X0[:,1], c="blue", label='0')
#     plt.scatter(X1[:,0], X1[:,1], c="red", label='1')
#
#     plt.legend()
#     plt.show()


def Classification(input_file = "input3.csv", output_file = "output3.csv"):
    # read file
    data = np.genfromtxt(input_file, delimiter=',')
    n = len(data[0])-1
    data = data[1:, :]
    X = data[:, :2]
    y = data[:, 2]
    # Plot data
    X0 = data[data[:, 2] == 0][:, :]
    X1 = data[data[:, 2] == 1][:, :]
    #plot_data(X0, X1)

    # create training and test sets
    X0_training, X0_test = train_test_split(X0, train_size = 0.6, shuffle=True)
    X1_training, X1_test = train_test_split(X1, train_size = 0.6, shuffle=True)

    data_training = np.vstack((X0_training, X1_training))
    #print(data_training)
    data_test = np.vstack((X0_training, X1_test))

    y_train = data_training[:, n]
    X_train = data_training[:, :n]
    y_test = data_test[:, n]
    X_test= data_test[:, :n]

    X_train, y_train = shuffle(X_train, y_train, random_state = 8)
    X_test, y_test = shuffle(X_test, y_test, random_state = 8)
    #print(y_train)
    #print(y_test)

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.60)
    #print(y_train)
    #print(y_test)
    #print(list(data_training[:,n]).count(0) / list(data_training[:,n]).count(1))
    #print(list(data_test[:,n]).count(0) / list(data_test[:,n]).count(1))
    #print(X)
    #print(y)

    # Classification methods
    #lkernel = SVMwithLinearKernel(X_train, y_train, X_test, y_test)
    #polykernel = SVMwithPolynomialKernel(X_train, y_train, X_test, y_test)
    #rbfkernel = SVMwithRBFKernel(X_train, y_train, X_test, y_test)
    lregression = LogisticRegression(X_train, y_train, X_test, y_test)
    #knn = KNearestNeighbors(X_train, y_train, X_test, y_test)
    #dt = DecisionsTrees(X_train, y_train, X_test, y_test)
    #rforest = RandomForest(X_train, y_train, X_test, y_test)

    results = [["svm_linear", lkernel],
               ["svm_polynomial", polykernel],
               ["svm_rbf", rbfkernel],
               ["logistic", lregression],
               ["knn", knn],
               ["decision_tree", dt],
               ["random_forest", rforest]]

    # Write results
    write_results(output_file, results)


def write_results(output_file, results):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for r in results:
            row = [r[0], r[1][0], r[1][1]]
            writer.writerow(row)


def SVMwithLinearKernel(X_train, y_train, X_test, y_test):
    parameters = {'kernel':['linear'], 'C': [0.1, 0.5, 1, 5, 10, 50, 100]}
    svr = svm.SVC()
    clf = GridSearchCV(svr, parameters)
    clf.fit(X_train, y_train)
    bestScore = clf.best_score_
    print(clf.best_score_)
    scores = cross_val_score(clf, X_test, y_test, cv=5)
    testScore = mean(scores)
    print(mean(scores))
    return bestScore, testScore


def SVMwithPolynomialKernel(X_train, y_train, X_test, y_test):
    parameters = {'kernel': ['poly'], 'C': [0.1, 1, 3], 'degree': [4, 5, 6], 'gamma': [0.1, 0.5]}
    svr = svm.SVC()
    clf = GridSearchCV(svr, parameters)
    clf.fit(X_train, y_train)
    bestScore = clf.best_score_
    print(clf.best_score_)
    scores = cross_val_score(clf, X_test, y_test, cv=5)
    testScore = mean(scores)
    print(mean(scores))
    return bestScore, testScore


def SVMwithRBFKernel(X_train, y_train, X_test, y_test):
    parameters = {'kernel': ['rbf'], 'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'gamma' : [0.1, 0.5, 1, 3, 6, 10]}
    svr = svm.SVC()
    clf = GridSearchCV(svr, parameters)
    clf.fit(X_train, y_train)
    bestScore = clf.best_score_
    print(clf.best_score_)
    scores = cross_val_score(clf, X_test, y_test, cv=5)
    testScore = mean(scores)
    print(mean(scores))
    return bestScore, testScore


def LogisticRegression(X_train, y_train, X_test, y_test):
    parameters = {'C': [0.1, 0.5, 1, 5, 10, 50, 100]}
    model = linear_model.LogisticRegression()
    clf = GridSearchCV(model, parameters)
    clf.fit(X_train, y_train)
    bestScore = clf.best_score_
    print(clf.best_score_)
    scores = cross_val_score(clf, X_test, y_test, cv=5)
    testScore = mean(scores)
    print(mean(scores))
    return bestScore, testScore


def KNearestNeighbors(X_train, y_train, X_test, y_test):
    parameters = { 'n_neighbors': [i for i in range(1,51)], 'leaf_size': [5*i for i in range(1,13)]}
    knn = neighbors.KNeighborsClassifier()
    clf = GridSearchCV(knn, parameters)
    clf.fit(X_train, y_train)
    bestScore = clf.best_score_
    print(clf.best_score_)
    scores = cross_val_score(clf, X_test, y_test, cv=5)
    testScore = mean(scores)
    print(mean(scores))
    return bestScore, testScore


def DecisionsTrees(X_train, y_train, X_test, y_test):
    parameters = { 'max_depth': [i for i in range(1,51)], 'min_samples_split': [2*i for i in range(1,11)]}
    dt = tree.DecisionTreeClassifier()
    clf = GridSearchCV(dt, parameters)
    clf.fit(X_train, y_train)
    bestScore = clf.best_score_
    print(clf.best_score_)
    scores = cross_val_score(clf, X_test, y_test, cv=5)
    testScore = mean(scores)
    print(mean(scores))
    return bestScore, testScore


def RandomForest(X_train, y_train, X_test, y_test):
    parameters = { 'max_depth': [i for i in range(1,51)], 'min_samples_split': [2*i for i in range(1,11)]}
    rf = ensemble.RandomForestClassifier()
    clf = GridSearchCV(rf, parameters)
    clf.fit(X_train, y_train)
    bestScore = clf.best_score_
    print(clf.best_score_)
    scores = cross_val_score(clf, X_test, y_test, cv=5)
    testScore = mean(scores)
    print(mean(scores))
    return bestScore, testScore


def main():
    input_file = sys.argv[1]

    output_file = sys.argv[2]

    Classification(input_file, output_file)


if __name__ == '__main__':
    main()