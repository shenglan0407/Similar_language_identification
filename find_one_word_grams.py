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
    for gram in one_grams:
        if gram in d.keys():
            #only recording 0 and 1 for wether a word is present or not
            pass
#             a = np.array([1],dtype = np.int8)
#             d[gram] += a[0]
        else:
            a = np.array([1],dtype=np.int8)
            d.update({gram:a[0]})
    return d,lg


f=open(os.getcwd()+'/partial_train_data/54k_sws_train.txt','r')


tic = time.clock()

dds= []
lgs = []
for line in f:
    d,lg = find_one_grams(line)
    with open(os.getcwd()+'/processed_training/sws_labels_54k.csv','a') as lg_f:
        csvwriter = csv.writer(lg_f, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow([lg])

    dds.append(d)
with open(os.getcwd()+'/processed_training/sws_54k.pkl', 'wb') as fp:
    pickle.dump(dds, fp)

toc = time.clock()
print "total time for process is %.2f seconds." % (toc-tic)

f.close()

