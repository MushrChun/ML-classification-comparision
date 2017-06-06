from utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from time import ctime
from time import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# stats arr for calculating average score over all folds
accuracyArr = np.zeros(10)
precisionArr = np.zeros(10)
recallArr = np.zeros(10)
f1Arr = np.zeros(10)

# accumulate the runtime of every fold
accTime = 0

# construct the RF classifier
es = RandomForestClassifier(max_features='sqrt', n_estimators=20, random_state=0, n_jobs = -1)

# define a stratified k-fold function to split 10 fold
skf = StratifiedKFold(n_splits=10)

fpr = dict()
tpr = dict()
roc_auc = dict()

# define color for draw plots
colors = ['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange', 'deeppink', 'navy', 'magenta', 'black']

# define plot parameters
plt.figure()
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic of Random Forest')
plt.plot([0, 1], [0, 1], 'k--', lw=2)

k = 0
for train, test in skf.split(X, y.ravel()):
    startTime = time()
    print('start ', k, 'fold at ', ctime(startTime))

    # training the RF model
    es.fit(X_scaled[train], y[train].ravel())

    # predict label of each test data
    predicted = es.predict(X_scaled[test])

    # get probability of target label
    predicted_prob = es.predict_proba(X_scaled[test])

    # get accuracy value
    accuracy = accuracy_score(y[test], predicted)

    # get precision,recall,fscore,support value
    precision, recall, fscore, support = precision_recall_fscore_support(y[test], predicted)

    # draw decent confusion matrix
    print(confusion_matrix(y[test], predicted))
    print(classification_report(y[test], predicted))

    # get average score based on all label types
    accuracyArr[k] = np.average(accuracy)
    precisionArr[k] = np.average(precision)
    recallArr[k] = np.average(recall)
    f1Arr[k] = np.average(fscore)

    y_bin = label_binarize(y[test], classes=[1, 2, 3, 4, 5, 6, 7])
    # calculate fpr and tpr value
    fpr[k], tpr[k], _ = roc_curve(y_bin.ravel(), predicted_prob.ravel())

    # calculate area under curve value
    roc_auc[k] = auc(fpr[k], tpr[k])

    # add new line of this loop
    plt.plot(fpr[k], tpr[k],
             label='micro-average ROC curve on fold %d (area = %0.2f)' % (k, roc_auc[k]),
             color=colors[k], linestyle='-', linewidth=2)

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


# draw the plot
plt.legend(loc="lower right")
plt.show()


