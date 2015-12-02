##############################################################################
# Copyright 2015 Stanford University and the Author
#
# Author: Shenglan Qiao
# 
# svm with LIBLINEAR, word one-gram
#
#############################################################################


##############################################################################
# Imports
##############################################################################
import pandas as pd
import numpy as np
import time
import os
import pickle
import csv

import sys
sys.path.insert(0, '/srv/zfs01/user_data/shenglan/liblinear-2.1/python')

import liblinear
from liblinearutil import *

##############################################################################
# Parameters
##############################################################################

# path to processed test data
test_examples = pickle.load(open(os.getcwd()+'/processed_test/sws_m2w_5400_test-gold.pkl', 'r'))

# path to file containing labels for test data
label_path = os.getcwd()+'/processed_test/sws_labels_test-gold.csv'

# load model to used (must match the one used to process test data)
m=load_model(os.getcwd()+'/trained_models/sws_m2w_5400.model')

# path to save confusion matrix
ConfMat_path = os.getcwd()+'/test_results/sws_m2w_5400_ConfMat_test-gold.csv'

# path to save predicted labels
predicted_labels_path = os.getcwd()+'/test_results/sws_m2w_5400_PredLab_test-gold.csv'
##############################################################################
# Code
##############################################################################
# test example labels labels
test_labels=[]
with open(label_path,'r') as f:
    csvreader = csv.reader(f, delimiter=' ', quotechar='|')
    for row in csvreader:
          test_labels.append(int(row[0]))
          
# predict using trained model
tests = [gen_feature_nodearray(this_example)[0] for this_example in test_examples]
predicted_labels = [liblinear.predict(m,x0) for x0 in tests]

# compute accuracy
correct = sum([1.0 for labels in zip(predicted_labels,test_labels) if labels[0]==labels[1]])
accuracy = correct/len(predicted_labels)

print "The accuracy from this classifier is %.2f" % accuracy

# make confusion matrix
confusion_matrix = np.zeros((3,3))
for labels in zip(predicted_labels,test_labels):
    confusion_matrix[labels[1]-1,labels[0]-1] += 1

with open(ConfMat_path, 'wb') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(['bs','hr','sr'])
    for row in confusion_matrix:
        csvwriter.writerow(row)
# write predicted_labels to file
with open(predicted_labels_path, 'wb') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(predicted_labels)