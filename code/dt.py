from utils import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from time import ctime
from time import time
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

# construct the DT classifier
dt = DecisionTreeClassifier(criterion='entropy', max_features=10, class_weight='balanced')

# stats arr for calculating average score over all folds
accuracyArr = np.zeros(10)
precisionArr = np.zeros(10)
recallArr = np.zeros(10)
f1Arr = np.zeros(10)

# accumulate the runtime of every fold
accTime = 0

# define a stratified k-fold function to split 10 fold
skf = StratifiedKFold(n_splits=10)
k = 0
for train, test in skf.split(X, y.ravel()):
    startTime = time()
    print('start ', k, 'fold at ', ctime(startTime))

    # training the DT model
    dt.fit(X_scaled[train], y[train].ravel())

    # predict label of each test data
    predicted = dt.predict(X_scaled[test])

    # draw decent confusion matrix
    print(confusion_matrix(y[test], predicted))

    # get probability of target label
    predicted_prob = dt.predict_proba(X_scaled[test])

    # get accuracy value
    accuracy = accuracy_score(y[test], predicted)

    # get precision,recall,fscore,support value
    precision, recall, fscore, support = precision_recall_fscore_support(y[test], predicted)

    print(classification_report(y[test], predicted))

    # get average score based on all label types
    accuracyArr[k] = np.average(accuracy)
    precisionArr[k] = np.average(precision)
    recallArr[k] = np.average(recall)
    f1Arr[k] = np.average(fscore)

    endTime = time()
    print('end ', k, 'fold at', ctime(endTime))
    interval = endTime - startTime
    accTime += interval
    k += 1

print('-------Average Score---------')
print('accuracy: ', np.average(accuracyArr))
print('precision: ', np.average(precisionArr))
print('recall: ', np.average(recallArr))
print('fscore: ', np.average(f1Arr))
print('accTime:', accTime)
