#!/usr/bin/env python
# coding: utf-8

# In[7]:


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


# In[ ]:




