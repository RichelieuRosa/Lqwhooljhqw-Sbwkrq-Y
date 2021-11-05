import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn import tree
from sklearn.svm import SVR
import pandas

#create panda dataframe

def cross_validation_3():
    import numpy as np
    from sklearn import datasets
    from sklearn import svm
    from sklearn.model_selection import KFold

    X,Y = datasets.load_wine(return_X_y=True)
    print("X.shape =", X.shape)
    print("Y.shape =", Y.shape)
    kf = KFold(n_splits=10)
    i = 0
    for train, test in kf.split(X):
        dft = pandas.DataFrame(columns = ['Data'])
        dftest = pandas.DataFrame(columns = ['Data'])
        p0 = list()
        p1 = list()
        for t in train:
            p0.append(t)
        dft['Data'] = p0
        dft.to_excel('Train'+str(i+1)+'.xlsx')
        for tst in test:
            p1.append(tst)
        dftest['Data'] = p0
        dftest.to_excel('Test'+str(i+1)+'.xlsx')

        i = i+1

def feature_elimination():

    X, Y= datasets.load_wine(return_X_y=True)
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, n_features_to_select=2, step=1)
    selector = selector.fit(X, Y)
    print("The mask of selected features: ", selector.support_)
    print("The feature ranking: ", selector.ranking_)

cross_validation_3()

feature_elimination()

## compute accuracy and F1 score using SVC

X, Y = datasets.load_wine(return_X_y = True)
classifier = svm.SVC()
scores = cross_val_score(classifier, X, Y, cv = 10)
print('Array of scores of the estimator for each run of the cross validation: ', scores)

f1scores = cross_val_score(classifier, X, Y, cv = 10, scoring = 'f1_macro')
print("f1_macro = ", f1scores)
