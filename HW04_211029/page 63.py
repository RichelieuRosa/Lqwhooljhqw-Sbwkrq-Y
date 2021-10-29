import numpy as np

def recursive_feature_elimination():
    from sklearn import datasets
    from sklearn.feature_selection import RFE
    from sklearn import tree
    from sklearn.svm import SVR
    
    X, Y= datasets.make_friedman1(n_samples=50, n_features=10, random_state=0)
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, n_features_to_select=5, step=1)
    selector = selector.fit(X, Y)
    print("The mask of selected features: ", selector.support_)
    print("The feature ranking: ", selector.ranking_)

recursive_feature_elimination()