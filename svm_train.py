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

# load training examples and their labels
train_data =  pickle.load(open(os.getcwd()+'/processed_training/sws_train_5400_bigram.pkl', 'r'))
train_labels = []
with open(os.getcwd()+'/processed_training/sws_labels_5400.csv','r') as f:
    csvreader = csv.reader(f, delimiter=' ', quotechar='|')
    for row in csvreader:
          train_labels.append(int(row[0]))

tic = time.clock()
# train svm with default settings in liblinear
prob=problem(train_labels,train_data)
m = train(prob)
save_model(os.getcwd()+'/trained_models/sws_bigram_5400.model',m)
toc = time.clock()
print 'Total time for training svm is %.2f seconds.' % (toc-tic)

