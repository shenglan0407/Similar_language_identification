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

# path to processed training data
train_data =  pickle.load(open(os.getcwd()+'/processed_training/sws_train_54k_bigram.pkl', 'r'))
# path to file containing labels for training data
label_path = os.getcwd()+'/processed_training/sws_labels_54k.csv'
# where to save the svm model
# naming convention: LanguageGroup_model_numberOfExamples.model.
# e.g.  sws_m2w_54k.model means southwest slavic group, modle with word bigrams
# and trained with 54k examples
save_path = os.getcwd()+'/trained_models/sws_m2w_54k.model'
##############################################################################
# Code
##############################################################################

# load training examples and their labels

train_labels = []
with open(label_path,'r') as f:
    csvreader = csv.reader(f, delimiter=' ', quotechar='|')
    for row in csvreader:
          train_labels.append(int(row[0]))

tic = time.clock()
# train svm with default settings in liblinear
prob=problem(train_labels,train_data)
m = train(prob)
save_model(save_path,m)
toc = time.clock()
print 'Total time for training svm is %.2f seconds.' % (toc-tic)

