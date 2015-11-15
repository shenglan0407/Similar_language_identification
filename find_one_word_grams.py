##############################################################################
# Copyright 2015 Stanford University and the Author
#
# Author: Shenglan Qiao
# 
# builds the sparse matrix for 1-word n-grams
#
#############################################################################


##############################################################################
# Imports
##############################################################################
import pandas as pd
import numpy as np
import time
import os
##############################################################################
# Code
##############################################################################


def find_one_grams(line):
    words = line.split()
    lg = words[-1]
    one_grams = words[:-1]
    d = {}
    for gram in one_grams:
        if gram in d.keys():
            a = np.array([1],dtype = np.int8)
            d[gram] += a[0]
        else:
            a = np.array([1],dtype=np.int8)
            d.update({gram:a[0]})
    return d,lg


f=open(os.getcwd()+'/2500_train.txt','r')
# f=open(os.getcwd()+'/test_test.txt','r')

tic = time.clock()

dds = pd.DataFrame()
lgs = []
for line in f:
    d,lg = find_one_grams(line)
    with open(os.getcwd()+'/lg_labels_2500.txt','a') as lg_f:
        lg_f.write(lg+'\n')
    dds = dds.append(d,ignore_index=True)
    
dds.to_csv(path_or_buf=os.getcwd()+'/one_word_grams_2500_train.txt',sep = ' ',index= False)
toc = time.clock()
print "total time for process is %.2f seconds." % (toc-tic)

f.close()

