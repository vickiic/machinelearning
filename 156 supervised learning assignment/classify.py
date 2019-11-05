#!/usr/bin/env python3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
names = ["KNN", "Linear SVM(bestmodel)", "Naive Bayes","LDA"]
classifiers = [
	KNeighborsClassifier(1), #best at 1 but has noise
	SVC(kernel="linear", C=60),
	GaussianNB(),
    LinearDiscriminantAnalysis()
    ]

train_data = np.load('X_v2.npy')
feature_names = train_data[0]
class_label = np.load('Y_v2.npy')
df_train = pd.read_csv('Y_v2.csv')
labels = pd.Series(class_label)
print(labels.value_counts())

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Remove 'id' and 'target' columns
X = train_data
y = class_label


# from sklearn.metrics import confusion_matrix
# from matplotlib import pyplot as plt

# conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
# print('Confusion matrix:\n', conf_mat)

# labels = ['Class 0', 'Class 1']
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
# fig.colorbar(cax)
# ax.set_xticklabels([''] + labels)
# ax.set_yticklabels([''] + labels)
# # plt.xlabel('Predicted')
# # plt.ylabel('Expected')
# # plt.show()
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X = pca.fit_transform(X)

plot_2d_space(X, y, 'Imbalanced dataset (2 PCA components)')

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X, y)
pass
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_resampled, y_resampled, test_size = 0.2, random_state=0)

from imblearn.combine import SMOTETomek

smt = SMOTETomek(ratio='auto')
X_smt, y_smt = smt.fit_sample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#i
#iterate over classifiers
for name, clf in zip(names, classifiers):
    clf.fit(X_resampled, y_resampled)
    testdata_accuracy = clf.score(X_test2, y_test2)
    traindata_accuracy = clf.score(X_train2, y_train2)
    print(name + ' test data accuracy: ' + ('%.9f' % testdata_accuracy).lstrip('0') + '\n')
    print(name + ' train data accuracy: ' + ('%.9f' % traindata_accuracy).lstrip('0') + '\n')