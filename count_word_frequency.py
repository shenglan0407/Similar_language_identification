##############################################################################
# Copyright 2015 Stanford University and the Author
#
# Author: Shenglan Qiao
# 
# This script takes the training corpus with digits and punctuations removed and
# all letters converted to lower case. Then it store the words and the word
# counts for different language in a h5py data structure. Each language is a
# new group in the h5py file and each word in a dataset in the language group.
# the word count is the first (0-indexed) element in each word dataset. The fraction
# by which the word appears in the entire corpus for a particular language is the 
# second element in the word dataset
#
#############################################################################


##############################################################################
# Imports
##############################################################################

import os
import h5py
import numpy as np


##############################################################################
# Code
##############################################################################

f_corp = open(os.getcwd()+'/data/data_no_punc_no_digit_lowercase_train.txt','r')
f_count = h5py.File(os.getcwd()+'/data/word_count_train.hdf5','a')

for line in f_corp:
    words = line.split()
    lg = words[-1]
    words = words[0:-1]
    if (lg in f_count)!= True:
        lg_gourp = f_count.create_group(lg)
    for word in words:
        if word in f_count[lg]:
            f_count[lg][word][0] +=1
        else:
            f_count[lg].create_dataset(word,data=[1.0,0.0],dtype = 'float')
f_corp.close()

for lg in f_count.keys():
    all_words = f_count[lg].keys()
    counts = [f_count[lg][word][0] for word in all_words]
    n_words = np.sum(counts)
    
    for word in all_words:
        f_count[lg][word][1] = f_count[lg][word][0]/float(n_words)

f_count.close()


