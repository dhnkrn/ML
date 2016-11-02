#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn.svm import SVC
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

clf = SVC(kernel='rbf', C=10000)

t0 = time()
clf.fit(features_train, labels_train)
print "training took: ", time() - t0,"s"

t1 = time()
pred = clf.predict(features_test)
print "prediction took: ", time() - t1,"s"
"""
print "10= ",pred[10]
print "26= ",pred[26]
print "50= ",pred[50]
"""
l = pred.tolist()
print l.count(1)

from sklearn.metrics import accuracy_score
print accuracy_score(labels_test, pred)

"""
for c in xrange(1000,30000, 500):
   clf  = SVC(kernel = 'rbf', C=c)
   clf.fit(features_train, labels_train)
   pred = clf.predict(features_test)
   print "C=",c," ", accuracy_score(labels_test, pred)
"""


#########################################################


