# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:05:23 2020

@author: Emanuel Marques Lourenco
"""
import numpy as np
import math
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression 

def resize(data,size):
    """
    Resizes an array 

    Parameters
    ----------
    :param data: a number array.
    :param size: the desired size for the returned array

    Returns
    ----------
    :return: array of size given by parameter.
    """

    sample_size = float(len(data))/size
    return [data[int(math.floor(i*sample_size))] for i in range(size)]


def classify(classifier_algorithm,classifier_name):
    """
    Applying classifiers and testing/evaluating results using k-fold cross
    validation

    Parameters
    ----------
    :param classifier_algorithm: The class of the classifier/algorithm to train
    the dataset on
    :param classifier_name: A string of the classifier name, purely for 
    better printing results on the console
    """
    #Setting up k-fold coss validation (number of folds can be changed)
    folds = 5  
    kf = KFold(n_splits=folds,shuffle=True, random_state=1)
    kf.get_n_splits(instances)
    
    added_accuracy = []
    added_recall = []
    added_precision = []
    added_gmean = []
    added_fscore = []
    added_fpr = []
  
    print('\nTraining classifier')
    for train_index, test_index in kf.split(instances):
        train_instances, test_instances = instances[train_index], instances[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]
        
        #initiating and training classifier   
        classifier = classifier_algorithm
        


        
        #training classifier
        classifier.fit(train_instances,train_labels)
        predicted_labels = classifier.predict(test_instances)
        
        #getting confusion matrix
        tn, fp, fn, tp = confusion_matrix(test_labels,predicted_labels).ravel()
        print('\nTP:', tp,'FP:',fp,'TN',tn,'FN:',fn)
        
        #Computing evaluation metrics in %
        accuracy = ((tp+tn) / (tp+tn+fp+fn))
        precision = (tp / (tp+fp))
        recall = (tp / (tp+fn))
        gmean = math.sqrt((tp/(tp+fn)*(tn/(tn+tp))))
        fscore = 2 * ((precision*recall)/(precision+recall))
        fpr = fp/(fp+tn)
        
        added_accuracy.append(accuracy)
        added_precision.append(precision)
        added_recall.append(recall)
        added_gmean.append(gmean)
        added_fscore.append(fscore)
        added_fpr.append(fpr)

        print('F-Score',fscore)
        print('G-mean:',gmean)
        print('Precision:',precision)
        print('Recall:',recall)
        print ('Accuracy:', accuracy)
        print('FPR',fpr)
            
    #printing average results
    print('\nResults over',folds,'folds: for the classifier', classifier_name)
    print('Average F-Score:',np.mean(added_fscore))
    print('Average g-mean:', np.mean(added_gmean))
    print('Average precision:', np.mean(added_precision))
    print('Average recall:', np.mean(added_recall))
    print('Average accuracy:', np.mean(added_accuracy))
    print('Average False Positive Rate:', np.mean(added_fpr))




#loading the profiles from both pulsar and non pulsar candidates 
pulsar_profiles = np.loadtxt('data/pulsar_profiles.csv', delimiter=',')
nonpulsar_profiles = np.loadtxt('data/nonpulsar_profiles.csv', delimiter=',')

#Creating pulsar and non-pulsar instances and respective labels
print('\nCreating pulsar features')
instances = []
labels = []
for i in range(len(pulsar_profiles)):
    instances.append(resize(pulsar_profiles[i],32))
    labels.append(1)
    
for i in range(len(nonpulsar_profiles)):
    instances.append(resize(nonpulsar_profiles[i],32))
    labels.append(0)
    



#needs to be converted into np array before fed into the algorithms
instances = np.asarray(instances)
labels = np.asarray(labels)

#Standardising the dataset
standardise = StandardScaler()
instances_std = standardise.fit_transform(instances)

#check if dataset has been properly standardised
print(np.mean(instances_std, axis=0))
print(np.std(instances_std, axis=0))


svc_classifier = SVC()
naiveb_classifier = GaussianNB()
randomfor_classifier = RFC(criterion='entropy',n_jobs = 2,n_estimators=100)
#solver parameter works better with lbfgs instead of adam (tested beforehand)
ann_classifier = MLPClassifier(solver='lbfgs')
tree_classifier = tree.DecisionTreeClassifier()
lr_classifier = LogisticRegression()

classify(svc_classifier,'Support Vector Machines')
classify(naiveb_classifier,'Naive Bayes')
classify(randomfor_classifier,'Random Forest')
classify(ann_classifier,'Multi-layer Perceptron')
classify(tree_classifier,'Decision Tree')
classify(lr_classifier,'Logistic Regression')
