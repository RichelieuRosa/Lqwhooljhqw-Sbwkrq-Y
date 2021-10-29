#!/usr/bin/env python
# coding: utf-8

# In[8]:


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


# In[ ]:




