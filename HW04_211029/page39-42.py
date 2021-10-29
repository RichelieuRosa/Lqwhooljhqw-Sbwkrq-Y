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