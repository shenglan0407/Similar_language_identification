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
# Code
##############################################################################

# load model
m=load_model(os.getcwd()+'/trained_models/sws_bigram_5400.model')

#load test examples and their labels
test_examples = pickle.load(open(os.getcwd()+'/processed_test/sws_m2w_5400_test-gold.pkl', 'r'))
test_labels=[]
with open(os.getcwd()+'/processed_test/sws_labels_test-gold.csv','r') as f:
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