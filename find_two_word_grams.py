##############################################################################
# Copyright 2015 Stanford University and the Author
#
# Author: Shenglan Qiao
# 
# builds the sparse matrix for 2-word n-grams
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


def find_two_grams(line):
    words = line.split()
    lg = words[-1]
    words = words[:-1]
    two_grams = ['%s %s'%(words[ii],words[ii+1]) for ii in range(len(words)-1)]
    two_grams.append('#begin#' + words[0])
    two_grams.append( words[-1]+'#end#')
    d = {}
    for gram in two_grams:
        if gram in d.keys():
            d[gram] += 1
        else:
            d.update({gram:1})
    return d,lg

f=open(os.getcwd()+'/2500_train.txt','r')
# f=open(os.getcwd()+'/test_test.txt','r')

tic = time.clock()

dds = pd.DataFrame()
lgs = []

for line in f:
    d,lg = find_two_grams(line)
    dds = dds.append(d,ignore_index=True)
    with open(os.getcwd()+'/two_gram_lg_labels_2500.txt','a') as lg_f:
        lg_f.write(lg+'\n')
    
dds.to_csv(path_or_buf=os.getcwd()+'/two_word_grams_2500_train.txt',sep = ' ',index= False)
toc = time.clock()
print "total time for process is %.2f seconds." % (toc-tic)

f.close()
