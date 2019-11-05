import numpy as np
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek

def mymodel(Xt, X, Y):
    Yp = np.ones(Xt.shape[0])
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    smt = SMOTETomek(ratio='auto')
    X_smt, y_smt = smt.fit_sample(X_resampled, y_resampled)
    plot_2d_space(X_smt, y_smt, 'SMOTE + Tomek links')
    clf = GaussianNB()
    model = clf.fit(X_smt,y_smt)
    Yp = model.predict(Xt)
    return Yp