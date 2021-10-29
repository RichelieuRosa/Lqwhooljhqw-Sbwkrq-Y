# page 51
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

X, Y = datasets.load_iris(return_X_y = True)
print("X.shape = ", X.shape)
print("Y.shape = ", Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4, random_state = 0)
print("X_train.shape = ", X_train.shape)
print("Y_train.shape = ", Y_train.shape)
print("X_test.shape = ", X_test.shape)
print("Y_test.shape = ", Y_test.shape)

classifier = svm.SVC().fit(X_train, Y_train)
acc = classifier.score(X_test, Y_test)
print("Accuracy =", acc)


# In[]
# page 52
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import metrics

X, Y = datasets.load_iris(return_X_y = True)
classifier = svm.SVC()
scores = cross_val_score(classifier, X, Y, cv = 3)
print('Array of scores of the estimator for each run of the cross validation: ', scores)


X, Y = datasets.load_iris(return_X_y = True)
classifier = svm.SVC()
scores = cross_val_score(classifier, X, Y, cv = 5, scoring = 'f1_macro')
print("f1_macro = ", scores)


# In[]
# page 54
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import KFold

X, Y = datasets.load_iris(return_X_y = True)
print("X.shape = ", X.shape)
print("Y.shape = ", Y.shape)
kf = KFold(n_splits = 5)
i = 0
for train, test in kf.split(X):
    print(str(i + 1) + "-th split:")
    print("Left margin of train set = ", train[0])
    print("Right margin of train set = ", train[len(train) - 1])
    print("Left margin of test set = ", test[0])
    print("Right margin of test set = ", test[len(test) - 1])    


