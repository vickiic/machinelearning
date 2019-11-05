import numpy as np
import pandas as pd
import csv
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture

names = ["K-Means", "Affinity Propagation", "Spectral Clustering","Mean Shift","Agglomerative Clustering","DBSCAN","Birch"]

clusters = [
    KMeans(n_clusters=7, random_state=1),
    AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity='euclidean', verbose=False),
    SpectralClustering(n_clusters=7,
             assign_labels="discretize",
             random_state=1),
    MeanShift(bandwidth=2),
    AgglomerativeClustering(n_clusters=7, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', pooling_func='deprecated'),
    DBSCAN(eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None),
    Birch(threshold=0.5, branching_factor=50, n_clusters=7, compute_labels=True, copy=True)
    ]

#read and create features & labels variables
data = pd.read_csv('glass_data_labeled.csv')
X = data[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]
y = data['Type']
# print(X)
# print(y)


for name, cl in zip(names, clusters):
    labels = cl.fit(X).labels_
    score = metrics.fowlkes_mallows_score(labels, y)
    print(name + ': ' + ('%.9f' % score).lstrip('0') + '\n')

pca = decomposition.PCA(n_components=6)
Xnew = pca.fit(X)
Xnew= pca.transform(X)
var_exp = pca.explained_variance_ratio_
print(pca.singular_values_)
cum_var_exp = np.cumsum(var_exp)

# Cumulative variance explained
for i, sum in enumerate(cum_var_exp):
    print("PC" + str(i+1), "Cumulative variance: %.3f% %" %(cum_var_exp[i]*100))

for name, cl in zip(names, clusters):
    labels = cl.fit(Xnew).labels_
    score = metrics.fowlkes_mallows_score(labels, y)
    print(name + ': ' + ('%.9f' % score).lstrip('0') + '\n')








