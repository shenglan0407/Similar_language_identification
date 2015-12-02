##############################################################################
# Copyright 2015 Stanford University and the Author
#
# Author: Shenglan Qiao
# 
# finds word 1-gram 
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

with open(os.getcwd()+'/processed_training/sws_train_5400_1gram_vocab_reduced.csv', 'r') as csvfile:
    for row in csvfile:
        word = row.decode('utf-8').split()[0]
        vocab.append(word)

def find_one_grams(line):
    words = line.split()
    lg = words[-1]
    if lg == all_lg[0]:
        lg = 1
    elif lg == all_lg[1]:
        lg = 2
    elif lg== all_lg[2]:
        lg = 3
    
    one_grams = words[:-1]
    d = {}
    # temp keep track of one-grams we have already seen
    temp = []
    for gram in one_grams:
        if gram in temp:
            # only recording 0 and 1 for wether a word is present or not
            pass
        else:
            temp.append(gram)
            number = int(np.where(np.array(vocab)==gram)[0][0]+1)
            a = np.array([1],dtype=np.int8)
            d.update({number:a[0]})
           
    return d,lg


f=open(os.getcwd()+'/partial_train_data/5400_sws_train.txt','r')
# f=open(os.getcwd()+'/partial_train_data/test.txt','r')

tic = time.clock()

dds= []
lgs = []
for line in f:
    d,_ = find_one_grams(line)
    dds.append(d)

with open(os.getcwd()+'/processed_training/sws_train_5400_reduced_vocab.csv','a') as lg_f:
        csvwriter = csv.writer(lg_f, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for this_w in vocab:
            csvwriter.writerow([this_w])

with open(os.getcwd()+'/processed_training/sws_train_54k.pkl', 'wb') as fp:
    pickle.dump(dds, fp)

toc = time.clock()
print "total time for process is %.2f seconds." % (toc-tic)

f.close()

