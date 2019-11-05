#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

h = .02  # step size in the mesh

names = ["KNN", "Linear SVM(bestmodel)", "Naive Bayes","LDA"]
classifiers = [
	KNeighborsClassifier(1), #best at 1 but has noise
	SVC(kernel="linear", C=60),
	GaussianNB(),
    LinearDiscriminantAnalysis()
    ]

zoo = np.load('pokemon_zoo_V1.npy')
test = np.load('pokemon_test_V1.npy')
feature_names = ['Gender','Weight', 'Height', 'HP', 'ATK', 'DEF', 'SpATK', 'SpDef', 'SPD']
X = np.array(zoo[feature_names].tolist()) #There must be a better way to fix this
y = zoo['Name']

linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable]

# iterate over datasets
X = [list(a) for a in X]
 # preprocess dataset, split into training and test part
X_train, X_test, y_train, y_test = \
   train_test_split(X, y, test_size=.4, random_state=42)

i=1
# iterate over classifiers
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    testdata_accuracy = clf.score(X_test, y_test)
    traindata_accuracy = clf.score(X_train, y_train)
    print(name + ' test data accuracy: ' + ('%.9f' % testdata_accuracy).lstrip('0') + '\n')
    print(name + ' train data accuracy: ' + ('%.9f' % traindata_accuracy).lstrip('0') + '\n')
    xnew = np.array(test[feature_names].tolist())
    ynew = clf.predict(xnew)
    print(ynew)
    np.save(name, ynew)
