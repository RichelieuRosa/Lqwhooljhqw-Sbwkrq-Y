#!/usr/bin/env python
# coding: utf-8

# In[7]:


def cross_validation_3():
    import numpy as np
    from sklearn import datasets
    from sklearn import svm
    from sklearn.model_selection import KFold

    X,Y = datasets.load_iris(return_X_y=True)
    print("X.shape =", X.shape)
    print("Y.shape =", Y.shape)
    kf = KFold(n_splits=5)
    i = 0
    for train, test in kf.split(X):
        print(str(i+1) + "-th split:")
        print("Left margin of train set = ", train[0])
        print("Right margin of train set = ", train[len(train)-1])
        print("Left margin of test set = ", test[0])
        print("Right margin of test set = ", test[len(test)-1])
        i = i+1


# In[11]:


def stratified_Kfold_1():
    import numpy as np
    from sklearn import datasets
    from sklearn import svm
    #from sklearn.model_selection import KFold
    from sklearn.model_selection import StratifiedKFold
    
    X = np.array([[1,2],[3,4],[1,2],[5,4]])
    Y = np.array([0,0,1,1])
    kf = StratifiedKFold(n_splits=2)   
    i = 0
    for train, test in kf.split(X,Y):
        print(str(i+1) + "-th split:")
        print("Train set: ")
        for i in range(len(train)):
            print(str(X[train[i]])+", "+str(Y[train[i]]))
        print("Test set: ")
        for i in range(len(test)):
            print(str(X[test[i]])+", "+str(Y[test[i]]))
        i = i+1
    


# In[15]:


def LabelEncoder():
    from sklearn import preprocessing
    City = ["paris","paris","tokyo","amsterdam"]
    le = preprocessing.LabelEncoder()
    le.fit(City)
    print("Class list: ", le.classes_)
    print("Encoded labels: ", le.transform(City))


# In[16]:


LabelEncoder()


# In[ ]:




