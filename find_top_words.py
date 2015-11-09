##############################################################################
# Copyright 2015 Stanford University and the Author
#
# Author: Shenglan Qiao
# 
# This script takes the word count data and identify the most frequently occuring
# word for each language. 
#
#############################################################################


##############################################################################
# Imports
##############################################################################
import h5py
import os
import numpy as np
import heapq
import csv

##############################################################################
# Code
##############################################################################

def find_top_words(data_base,lg,n_top = 1000):
    #print lg
    all_words = data_base[lg].keys()
    #print all_words[0]
    counts = [data_base[lg][word][0] for word in all_words]
    max_inds = arg_maxN(counts,n_top)
    top_words = [all_words[i] for i in max_inds]
    top_freqs = [data_base[lg][word][1] for word in top_words]
    
    return top_words,top_freqs
    
def arg_maxN(a,N):
    return np.argsort(a)[::-1][:N]

f = h5py.File(os.getcwd()+'/data/word_count_train.hdf5','r')

with open(os.getcwd()+'/data/top_words_train.csv', 'wb') as csvfile:
    for lg in f.keys():
        ws,ff = find_top_words(f,lg)
        ws=[item.encode('utf-8') for item in ws]
        csvwriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow([lg])
        csvwriter.writerow(ws)
        csvwriter.writerow(ff)
f.close()