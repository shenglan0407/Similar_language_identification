##############################################################################
# Copyright 2015 Stanford University and the Author
#
# Author: Shenglan Qiao
# 
# finds word bigrams
#
#############################################################################


##############################################################################
# Imports
##############################################################################
import numpy as np
import time
import os
import pickle
import csv
##############################################################################
# Code
##############################################################################


all_lg = ['bs','hr','sr']
vocab=[]

def find_two_grams(line):
    words = line.split()
    
    # assign language label
    lg = words[-1]
    if lg == all_lg[0]:
        lg = 1
    elif lg == all_lg[1]:
        lg = 2
    elif lg== all_lg[2]:
        lg = 3
    
    # list all word bigrams in the example
    words = words[:-1]
    two_grams = ['%s %s'%(words[ii],words[ii+1]) for ii in range(len(words)-1)]
    two_grams.append('#begin#' + words[0])
    two_grams.append( words[-1]+'#end#')
    
    # create sparse formate dictionary 
    d = {}
    
    # temp keep track of one-grams we have already seen
    temp = []
    for gram in two_grams:
        if gram in temp:
            # only recording 0 and 1 for wether a word is present or not
            pass
        else:
            temp.append(gram)
            if gram in vocab:
                number = int(np.where(np.array(vocab)==gram)[0][0]+1)
                a = np.array([1],dtype=np.int8)
                d.update({number:a[0]})
            else:
                vocab.append(gram)
                number = len(vocab)
                a = np.array([1],dtype=np.int8)
                d.update({number:a[0]})
    return d,lg

f=open(os.getcwd()+'/partial_train_data/54k_sws_train.txt','r')
# f=open(os.getcwd()+'/partial_train_data/test.txt','r')
tic = time.clock()

dds= []
for line in f:
    d,_ = find_two_grams(line)
    dds.append(d)

with open(os.getcwd()+'/processed_training/sws_train_54k_bigram_vocab.csv','a') as lg_f:
        csvwriter = csv.writer(lg_f, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for this_w in vocab:
            csvwriter.writerow([this_w])

with open(os.getcwd()+'/processed_training/sws_train_54k_bigram.pkl', 'wb') as fp:
    pickle.dump(dds, fp)
    
toc = time.clock()
print "total time for process is %.2f seconds." % (toc-tic)

f.close()
