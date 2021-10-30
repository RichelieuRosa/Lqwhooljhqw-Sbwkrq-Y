# Page 1 - 30

import numpy as np

def create_ndarray_object():
    arr_from_list = np.array([1,2,3,4,5])
    arr_from_tuple = np.array((1,2,3,4,5))
    print("arr_from_list = ",arr_from_list)
    print("arr_from_tuple = ",arr_from_tuple)
    print("type(arr_from_list) = ",type(arr_from_list))
    print("type(arr_from_tuple) = ",type(arr_from_tuple))
    
create_ndarray_object()

def dimensions_in_arrays():
    arr_0D = np.array(5)
    arr_1D = np.array([1,2,3,4,5])
    arr_2D = np.array([[1,2,3],[4,5,6]])
    print("arr_0D = ", arr_0D)
    print("arr_1D = ", arr_1D)
    print("arr_2D = ", arr_2D)
    
    print("Demension of arr_0D = ",arr_0D.ndim)
    print("Demension of arr_1D = ",arr_1D.ndim)
    print("Demension of arr_2D = ",arr_2D.ndim)

dimensions_in_arrays()

def indexing_and_slicing():
    arr_1D = np.array([1,2,3,4])
    arr_2D = np.array([[1,2,3,4,5],[6,7,8,9,10]])
    print("1st element = ", arr_1D[0])
    print("2nd element on 1st dimension = ", arr_2D[1,3])
    
    #1-D Array
    arr_1D = np.array([1,2,3,4,5,6,7,8,9,10])
    print("Slice lements from index 1 to index 5 = ", arr_1D[1:5])
    print("Slice lements from index 3 to the end = ", arr_1D[3:])
    print("Use the step value to return every other element from index 1 to index 5 = ", arr_1D[1:5:2])
    print("Return every other element from the entire array: ", arr_1D[::2])
    
    #2-D Array
    arr_2D = np.array([[1,2,3,4,5],[6,7,8,9,10]])
    print("Slice index 1 to index 4 (not incluede)from the second element", arr_2D[1,1:4])
    print("From both elements, return index 2: ", arr_2D[0:2,2])
    print("From both elements, slice index 1 to index 4 (not included): ",arr_2D[0:2,1:4])
    
indexing_and_slicing()

def checking():
    arr_integer = np.array([1,2,3,4])
    arr_string = np.array(['apple','bnana','cherry'])
    
    print(arr_integer.dtype)
    print(arr_string.dtype)
    
checking()

def creating():
    #Create an array with data type string:
    arr_S = np.array([1,2,3,4], dtype='S')
    
    #Create an array with data type 4 bytes integer:
    arr_i4 = np.array([1,2,3,4],dtype='i4')
    
    print(arr_S)
    print(arr_S.dtype)
    print(arr_i4)
    print(arr_i4.dtype)
    
creating()

def converting_datatype():
    arr1 = np.array([1.1,2.1,3.1])
    #Change data type from float to integer by using 'i' as parameter value:
    newarr11 = arr1.astype('i')
    #Change data type from float to integer by using int as parameter value:
    newarr12 = arr1.astype(int)

    #Change data type from integer to boolean:
    arr2 = np.array([1,0,3])
    newarr2 = arr2.astype(bool)
    
    print(newarr11)
    print(newarr11.dtype)
    print(newarr12)
    print(newarr12.dtype)
    print(newarr2)
    print(newarr2.dtype)

converting_datatype()

def copy_function():
    arr = np.array([1,2,3,4,5])
    arr_copy = arr.copy()
    arr[0] = 10
    
    print("Original array: ", arr)
    print("Copy array: ", arr_copy)

copy_function()

def shape_function():
    arr = np.array([[1,2,3,4],[5,6,7,8]])
    print("Shape of array = ", arr.shape)
    
shape_function()

def reshape_function():
    arr = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
    newarr = arr.reshape(4,3)
    print(newarr)

reshape_function()

def interating():
    
    arr = np.array([[1,2,3],[4,5,6]])
    
    print("Method 1:")
    for x in arr:
        for y in x:
            print(y)
            
    print("-----")
    print("Method 2:")
    for x in np.nditer(arr):
        print(x)
    for idx,x in np.ndenumerate(arr):
        print(idx,x)
        
interating()

def join_function():
    print("1-D array")
    arr1 = np.array([1,2,3])
    arr2 = np.array([4,5,6])
    arr = np.concatenate((arr1,arr2))
    print(arr)
    
    print("-----")
    print("2-D array")
    arr1 = np.array([[1,2],[3,4]])
    arr2 = np.array([[5,6],[7,8]])
    arr = np.concatenate((arr1,arr2),axis=1)
    print(arr)
    
    print("stack()")
    arr1 = np.array([1,2,3])
    arr2 = np.array([4,5,6])
    arr = np.stack((arr1,arr2),axis=1)
    
    print("Stacking Along Rows: hstack()")
    arr = np.hstack((arr1,arr2))
    print(arr)
    
    print("Stacking Along Columns: vstack()")
    arr = np.vstack((arr1,arr2))
    print(arr)
    
    print("Stack Along Height(depth): dstack()")
    arr = np.dstack((arr1,arr2))
    print(arr)
    
join_function()

def split_function():
    print("Split 1-D array in 3 parts: ")
    arr = np.array([1,2,3,4,5,6])
    newarr = np.array_split(arr,3)
    print(newarr)
    print(newarr[0])
    print(newarr[1])
    print(newarr[2])
    
    print("-----")
    arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
    print("Split the 2-D array into three 2-D arrays:")
    newarr = np.array_split(arr,3)
    print(newarr)
    print("Split the 2-D array into three 2-D arrays along rows:")
    newarr = np.array_split(arr,3,axis=1)
    print(newarr)

split_function()


def search_function():
    arr = np.array([1,2,3,4,5,4,4])
    print("The indexes where the value is 4: ")
    x= np.where(arr ==4)
    print(x)
    
    print("The indexes where the values are odd:")
    x = np.where(arr%2==1)
    print(x)
    
    arr = np.array([6,7,8,9])
    print("The indexes where the value 7 should be inserted: ")
    x = np.searchsorted(arr,7)
    print(x)
    
    print("The indexes where the value 7 should be inserted, starting from the right: ")
    x = np.searchsorted(arr,7,side='right')
    print(x)
    
    print("The indexes where multiple values should be inserted:")
    arr = np.array([1,3,5,7])
    x = np.searchsorted(arr,[2,4,6])
    print(x)
    
search_function()

def sort_function():
    print("Sort numeric array:")
    arr = np.array([3,2,0,1])
    print(np.sort(arr))
    
    print("Sort string array:")
    arr = np.array(['bnana','cherry','apple'])
    print(np.sort(arr))
    
    print("Sort boolean array:")
    arr = np.array([True, False, True])
    print(np.sort(arr))
    
    print("Sort 2-D array:")
    arr = np.array([[3,2,4],[5,0,1]])
    print(np.sort(arr))
    
sort_function()

from numpy import random
def random_numbers():
    print("Generate a random integer from 0 to 100: ")
    x = random.randint(100)
    
    print("Generate a random float from 0 to 1: ")
    x = random.rand()
    print(x)
    
    print("Generate a 1-D array containing 5 random integers from 0 to 100: ")
    x = random.randint(100,size=(5))
    print(x)
    
    print("Generate a 2-D array with 3 rows, each row containing 5 random integers from 0 to 100:")
    x = random.randint(100, size=(3,5))
    print(x)
    
    print("Generate a 1-D array containing 5 random floats:")
    x = random.rand(5)
    print(x)
    
    print("Generate a 2-D array with 3 rows, each row containing 5 random numbers: ")
    x = random.rand(3,5)
    print(x)
    
    x = random.choice([3,5,7,9])
    print("Return one of the values in an array: ", x)
    
    x = random.choice([3,5,7,9], size=(3,5))
    print("Generate a 2-D array that consists of the values in the array parameter (3,5,7, and 9): ",x)
    
random_numbers()

def random_permutations():
    arr = np.array([1,2,3,4,5])
    random.shuffle(arr)
    print(arr)
    arr = np.array([1,2,3,4,5])
    print(random.permutation(arr))

random_permutations()

import pandas as pd

def pandas_series():
    a = [1,10,5]
    SimpleSeries = pd.Series(a)
    print(SimpleSeries)
    print("1st value: ",SimpleSeries[0])
    print("2nd value: ",SimpleSeries[1])
    
    print("-----")
    NamedSeries = pd.Series(a, index = ["x","y","z"])
    print("Naming the labels: ")
    print(NamedSeries)
    print("The value of y : ", NamedSeries["y"])
    
pandas_series()

def pandas_dataframe():
    data = {"calories":[420,380,390],"duration":[50,40,45]}
    #load data into a DataFrame object:
    df = pd.DataFrame(data)
    print(df)
    print("Row 0 :")
    print(df.loc[0])
    print("Row 1 :")
    print(df.loc[1])
    print("Row 0 and 1 :")
    print(df.loc[[0,1]])
    #named indexes
    df = pd.DataFrame(data,index = ["day1","day2","day3"])
    print(df)
    print("day2: ")
    print(df.loc["day2"])
    
    df = pd.read_csv('data.csv')
    
    print("The first 5 rows: ")
    print(df.head())
    
    print("\n")
    print("The last 5 rows: ")
    print(df.tail())
    
    print("\n")
    print("Information about the data: ")
    print(df.info())
    
pandas_dataframe()

# page39 confusion matrix
from sklearn.metrics import confusion_matrix
y_actual = [2, 0, 2, 2, 0, 1]
y_prediction = [0, 0, 2, 2, 0, 2]
confusion_matrix(y_actual, y_prediction)

y_true = ["class1", "class2", "class1", "class3", "class2", "class1"]
y_pred = ["class2", "class2", "class1", "class3", "class1", "class1"]
confusion_matrix(y_true, y_pred)

# page40 TP, TN, FP, FN (1/2)
CM = confusion_matrix(y_true, y_pred)
TN = CM[0][0]
FN = CM[1][1]
TP = CM[1][1]
FP = CM[0][1]
TN, FP, FN, TP = confusion_matrix([0, 1, 1, 1], [1, 1, 1, 0]).ravel()
print(TN, FP, FN, TP)
# if you use pandas/numpy, you can do:
# FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
# FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
# TP = np.diag(confusion_matrix)
# TN = confusion_matrix.values.sum() - (FP + FN + TP)

# page41 TP, TN, FP, FN (2/2)
def MetricsComputation(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_pred)):
        if y_actual[i] == y_pred[i] == 1:
            TP += 1
        if y_pred[i] == 1 and y_actual[i] !=y_pred[i]:
            FP += 1
        if y_actual[i] == y_pred[i] == 0:
            TN += 1
        if y_pred[i] == 0 and y_actual[i] != y_pred[i]:
            FN += 1
    return (TP, FP, TN, FN)

# page42 Accuracy
from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3]
y_actual = [0, 1, 2, 3]
accuracy_score(y_actual, y_pred)

# Page 44- 47

from sklearn.metrics import f1_score
def scikit_F1():
    
    y_true=[0,1,2,0,1,2]
    y_pred=[0,2,1,0,0,1]
    F1_macro=f1_score(y_true,y_pred,average='macro')
    F1_micro=f1_score(y_true,y_pred,average='micro')
    F1_weighted=f1_score(y_true,y_pred,average=None)
    print('F1_macro',F1_macro)
    print('F1_micro',F1_micro)
    print('F1_weighted',F1_weighted)
    print('F1_none,F1_none')
scikit_F1()   


# In[10]:


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
def scikit_tpr():
    
    
    y_true=[0,1,2,0,1,2]
    y_pred=[0,2,1,0,0,1]

    precision_macro=precision_score(y_true,y_pred,average='macro')
    precision_micro=precision_score(y_true,y_pred,average='micro')
    precision_weighted=precision_score(y_true,y_pred,average='weighted')
    precision_none=precision_score(y_true,y_pred,average=None)

    recall_macro=recall_score(y_true,y_pred,average='macro')
    recall_micro=recall_score(y_true,y_pred,average='micro')
    recall_weighted=recall_score(y_true,y_pred,average='weighted')
    recall_none=recall_score(y_true,y_pred,average=None)

    print('Precision_macro=',precision_macro)
    print('Precision_micro=',precision_micro)
    print('Precision_weighted=',precision_weighted)
    print('Precision_none=',precision_none)
    print('-----')
    print('Recall_macro=',recall_macro)
    print('Recall_micro=',recall_micro)
    print('Recall_weighted=',recall_weighted)
    print('Recall_none=',recall_none)
scikit_tpr()


# In[13]:


import numpy as np
from sklearn import metrics
def scikit_AUCROC():
    y_actual=np.array([1,1,2,2])
    y_predict=np.array([0.1,0.4,0.35,0.8])
    FPR,TPR,thresholds=metrics.roc_curve(y_actual,y_predict,pos_label=2)
    auc=metrics.auc(FPR,TPR)
    print('AUC score=',auc)
    y_actual=np.array([1,1,2,2])
    y_scores=np.array([0.1,0.4,0.35,0.8])
    fpr,tpr,threshold=metrics.roc_curve(y_actual,y_scores,pos_label=2)
    print(fpr)
    print(tpr)
    print(thresholds)
scikit_AUCROC()

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

    
# Page 55- 60

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



def LabelEncoder():
    from sklearn import preprocessing
    City = ["paris","paris","tokyo","amsterdam"]
    le = preprocessing.LabelEncoder()
    le.fit(City)
    print("Class list: ", le.classes_)
    print("Encoded labels: ", le.transform(City))



LabelEncoder()

# Page 59 - 62
def LabelEncoder_function():
    from sklearn import preprocessing
    
    City = ["paris", "paris", "tokyo", "amsterdam"]
    le = preprocessing.LabelEncoder()
    le.fit(City)
    print("Class list", le.classes_)
    print("Encoded labels", le.transform(City))
    
LabelEncoder_function()


# In[1]:


def OneHotEncoder_function():
    from sklearn import preprocessing
    
    X = [['Male', 1], ['Female', 3], ['Female', 2]]
    enc = preprocessing.OneHotEncoder()
    enc.fit(X)
    print("The categories of each feature: ")
    print(enc.categories_)
    print("Encoded: ")
    print(enc.transform(X).toarray())
    
OneHotEncoder_function()


# In[2]:


def OrdinalEncoder_function():
    from sklearn.preprocessing import OrdinalEncoder
    
    enc = OrdinalEncoder()
    X = [['Male', 1], ['Female', 3], ['Female', 2]]
    enc.fit(X)
    print(enc.categories_)
    print(enc.transform([['Female', 3], ['Male', 1]]))
OrdinalEncoder_function()

# Page 63
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
