# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:48:15 2020

@author: Emanuel Marques Lourenco
"""


import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import sys
import math
import csv

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression




def centre_on_peak(data):
    """
    Centres a number array

    Parameters
    ----------
    :param data: a number array.

    Returns
    ----------
    :return: the same array with peak centred [mean, stdev, shew, kurtosis].
    """
    # Error handling
    if data is None or len(data) == 0:
        return []

    # First we need a minimum value to compare against.
    max_value_found = sys.float_info.min
    #index to find the location of the peak, and items for the size of the list
    index_of_max_value = -1
    items = int(len(data))
    """Update the index and the max value variables if the next value in the 
       array is larger, or equal to the max value found"""
    for i in range(0,items):
        if data[i]>= max_value_found:
            max_value_found = data[i]
            index_of_max_value = i
            
    """finding shifts needed to get the highest value in the centre"""
     
     # Check if the data is even
    if  items  % 2 == 0:
        #items is a float number, must be turned to integer first
        midpoint = int((items - 1) / 2) 
    else:
        midpoint = int(items / 2)
        
    #Counting the left shifts needed to centre the data.
    shift = 0
    # Updating the numbers of shifts based on the index of the max value.
    if index_of_max_value < midpoint:
        shift = (midpoint+index_of_max_value) + 1
    elif index_of_max_value == midpoint:
        return data # No need to shift, max value already in the middle.
    else:
        shift = index_of_max_value - midpoint
          
    """shifting the data"""
    for i in range(shift):
        #temporarily store the data on first position
        temp = data[0]
        #do the shift, every item will move one to the left
        for i in range(items-1):
            data[i] = data[i + 1] 
        #item on the first position gets stored now on the last position
        data[items-1] = temp
    #Plotting data to test if it is getting properly centered, (commented out)    
    #plt.plot(data)
    #plt.show()
    return data


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
    #resizes an array, new size speified by size parameter
    sample_size = float(len(data))/size
    return [data[int(math.floor(i*sample_size))] for i in range(size)]

def get_stats(data):  
    """
    Computes mean, standard deviation, skewness and kurtosis of an array

    Parameters
    ----------
    :param data: a number array.

    Returns
    ----------
    :return: computed values as an array [mean, stdev, skew, kurt].
    """
    #length of the array
    n = len(data)
    #calculating mean
    sum_data = sum(data)
    mean = sum_data / n
    k = 0
    for value in data:
        k = k + (value-mean)**2
    stdev = math.sqrt(k/len(data))
    kurt = stats.kurtosis(data)
    skew = stats.skew(data)

    return mean, stdev, skew, kurt
   
   
def get_features(profiles,snrvalues,dmvalues):
    """
    Computes the machine learning features with the 3 supplied data arrays, 9
    different features are computed however returned features can be changed 
    depending on feature selection, 

    Parameters
    ----------
    :param profiles: array of pulsar candidate profile.
    :param snrvalues: array of SNR values of the DM curve of the pulsar candidate
    :param dmvalues: array DM values of the DM curve of the pulsar candidate

    Returns
    ----------
    :return: The computed features as an array [mean_profiles, stdev_profiles, 
             skew_profiles, kurt_profiles, profiles_peak, snr_peak_dm_product,
             snr_gmean].
    """
    # Check data is not empty
    if profiles is None or len(profiles) == 0:  
        print('Could not load profile data')
        return [0,0,0,0,0,0]


    mean_profiles, stdev_profiles, skew_profiles, kurt_profiles = get_stats(profiles)
    
    #The highest value of the profile of a pulsar should be higher than 
    #the ones from noise?, so maybe it can be used as a
    #feature, preliminary tests, shows that this might be suitable
    profiles_peak = max(profiles)
    #same but for lowest value
    profiles_lowest = min(profiles) 
    
    snr_peak_index = np.where(snrvalues == np.amax(snrvalues))
    snr_peak = max(snrvalues)
    snr_peak_dm_product = snr_peak * dmvalues[snr_peak_index[0][0]]
    
    snr_lowest_index = np.where(snrvalues == np.amin(snrvalues))
    snr_lowest = min(snrvalues)
    snr_lowest_dm_product = snr_lowest * dmvalues[snr_lowest_index[0][0]]
    
    #removing zeroes from snr values so geometric mean can be computed
    snrvalues = np.trim_zeros(snrvalues)
    snr_gmean = stats.gmean(snrvalues)

    return [mean_profiles, stdev_profiles, skew_profiles, kurt_profiles, 
            profiles_peak, snr_peak_dm_product,snr_gmean, profiles_lowest,
            snr_lowest_dm_product]


def classify(classifier_algorithm,classifier_name,smote):
    """
    Applying classifiers and testing/evaluating results using k-fold cross
    validation

    Parameters
    ----------
    :param classifier_algorithm: The class of the classifier/algorithm to train
    the dataset on
    :param classifier_name: A string of the classifier name, purely for 
    better printing results on the console
    :param smote: boolean to decide whether smote treatment is used or not
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
        

        #standardising data
        scaler = StandardScaler()
        train_instances = scaler.fit_transform(train_instances)
        test_instances = scaler.transform(test_instances)
        
        #Applying Smote imbalance treatment
        if smote == True:
            print('Applying SMOTE Imbalance Treatment')
            sm = SMOTE(random_state=42)
            train_instances, train_labels = sm.fit_sample(train_instances, train_labels)
        
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
        """
        print('F-Score',fscore)
        print('G-mean:',gmean)
        print('Precision:',precision)
        print('Recall:',recall)
        print ('Accuracy:', accuracy)
        print('FPR',fpr)"""
            
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

#loading snr values from dm-curve
pulsar_snrvalues = []
nonpulsar_snrvalues = []
with open('data/pulsar_snrvalues.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        for i in range(len(row)):
            row[i] = float(row[i])
        pulsar_snrvalues.append(row)
with open('data/nonpulsar_snrvalues.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        for i in range(len(row)):
            row[i] = float(row[i])
        nonpulsar_snrvalues.append(row)

#loading dm values from dm-curve
pulsar_dmvalues = []
nonpulsar_dmvalues = []
with open('data/pulsar_dmvalues.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        for i in range(len(row)):
            row[i] = float(row[i])
        pulsar_dmvalues.append(row)
with open('data/nonpulsar_dmvalues.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        for i in range(len(row)):
            row[i] = float(row[i])
        nonpulsar_dmvalues.append(row)

""" This is just the profiles
#loading flattened subintegrations from data
pulsar_subintegrations = np.loadtxt('data/pulsar_subintegrations.csv', delimiter=',')
nonpulsar_subintegrations = np.loadtxt('data/nonpulsar_subintegrations.csv', delimiter=',')
"""


# Print overview details of the data we've obtained.
print ('Loaded pulsar profiles: ', len(pulsar_profiles),
       ', snr values:', len(pulsar_snrvalues),
       ', dm values:', len(pulsar_dmvalues))
print ('Loaded noise/RFI profiles: ', len(nonpulsar_profiles),
       ', snr values:', len(nonpulsar_snrvalues),
       ', dm values', len(nonpulsar_dmvalues))



#loop over each item in the data arrays to normalise and centre the peaks of 
#each profile (each item is a different profile)
for i in range(len(pulsar_profiles)):
    #pulsar_profiles[i] = standardise(pulsar_profiles[i])
    #should not affect statistical features in any way
    pulsar_profiles[i] = centre_on_peak(pulsar_profiles[i])
    

for i in range(len(nonpulsar_profiles)):
    #nonpulsar_profiles[i] = normalise standardise(nonpulsar_profiles[i])
    #should not affect statistical features in any way
    nonpulsar_profiles[i] = centre_on_peak(nonpulsar_profiles[i])
    
"""
 #Plotting examples
plt.plot(pulsar_dmvalues[198],pulsar_snrvalues[198], 'r')
plt.xlabel('DM')
plt.ylabel('S/N')
plt.title('Pulsar DM Curve')
plt.show()

plt.plot(nonpulsar_dmvalues[103],nonpulsar_snrvalues[103], 'b')
plt.xlabel('DM')
plt.ylabel('S/N')
plt.title('Non-pulsar DM Curve')
"""

#Creating pulsar and non-pulsar instances and respective labels
print('\nCreating pulsar features')
instances = []
labels = []
for i in range(len(pulsar_profiles)):
    
    instances.append(get_features(pulsar_profiles[i],pulsar_snrvalues[i],
                                  pulsar_dmvalues[i]))
    labels.append(1)
print('Creating non-pulsar features')
for i in range(len(nonpulsar_profiles)):
    
    instances.append(get_features(nonpulsar_profiles[i],nonpulsar_snrvalues[i],
                                   nonpulsar_dmvalues[i]))
    labels.append(0)


#needs to be converted into np array before fed into the algorithms
instances = np.asarray(instances)
labels = np.asarray(labels)



"""Saving the final dataset as a cvs file by having features, and their labels
as the last column of the array
print('Saving finished dataset')

finished_dataset = np.insert(instances,instances.shape[1],labels, axis=1)
with open('finished_dataset.csv', 'w') as output:
       writer = csv.writer(output, lineterminator='\n')
       for line in finished_dataset:
           writer.writerow(line) 
"""


"""
The parameter class_weight can penalise mistakes to the minority class
in order to mitigate the imbalance of the training dataset
include the argument probability=True if it is useful to enable 
probability estimates for SVM algorithms."""
svc_classifier = SVC()
naiveb_classifier = GaussianNB()
randomfor_classifier = RFC(criterion='entropy',n_jobs = 2,n_estimators=100)
#solver parameter works better with lbfgs instead of adam (tested beforehand)
ann_classifier = MLPClassifier(solver='lbfgs')

"""scikit-learn uses an optimised version of the CART algorithm; however, 
scikit-learn implementation does not support categorical variables for now.
CART (Classification and Regression Trees) is very similar to C4.5, but it 
differs in that it supports numerical target variables (regression) and does 
not compute rule sets. CART constructs binary trees using the feature and 
threshold that yield the largest information gain at each node."""
tree_classifier = tree.DecisionTreeClassifier()
lr_classifier = LogisticRegression()


classify(svc_classifier,'Support Vector Machines',smote=False)
classify(naiveb_classifier,'Naive Bayes',smote=False)
classify(randomfor_classifier,'Random Forest',smote=False)
classify(ann_classifier,'Multi-layer Perceptron',smote=False)
classify(tree_classifier,'Decision Tree',smote=False)
classify(lr_classifier,'Logistic Regression',smote=False)

classify(svc_classifier,'Support Vector Machines',smote=True)
classify(naiveb_classifier,'Naive Bayes',smote=True)
classify(randomfor_classifier,'Random Forest',smote=True)
classify(ann_classifier,'Multi-layer Perceptron',smote=True)
classify(tree_classifier,'Decision Tree',smote=True)
classify(lr_classifier,'Logistic Regression',smote=True)



#Testing different parameters of the SVC classifier
svc_classifier_one = SVC(kernel='linear')
svc_classifier_two = SVC(kernel='rbf')
svc_classifier_three = SVC(kernel='poly')
svc_classifier_four = SVC(kernel='sigmoid')

classify(svc_classifier_one,'SVC kernel=linear',smote=True)
classify(svc_classifier_two,'SVC kernel=rbf',smote=True)
classify(svc_classifier_three,'SVC kernel=polynomial',smote=True)
classify(svc_classifier_four,'SVC kernel=sigmoid',smote=True)




