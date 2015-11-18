##############################################################################
# Copyright 2015 Stanford University and the Author
#
# Author: Shenglan Qiao
# 
# process test set into format used by liblinear 
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

# load the vocabulary learnt from training set
with open(os.getcwd()+'/processed_training/sws_train_5400_vocab.csv', 'r') as f:
    vocab = []
    csvreader =  csv.reader(f, delimiter=' ', quotechar='|')
    for row in csvreader:
        vocab.append(row[0])

test_f=open(os.getcwd()+'/data/sws_test-gold.txt','r')

# language present in test set
all_lg = ['bs','hr','sr']

test_labels=[]
test_examples=[]

for line in test_f:

    count +=1
    line=line.split()
    words=line[:-1]
    lg=line[-1]

    if lg == all_lg[0]:
        lg = 1
    elif lg == all_lg[1]:
        lg = 2
    elif lg== all_lg[2]:
        lg = 3
    test_labels.append(lg)
    
    # write labels of each test example to file
    with open(os.getcwd()+'/processed_test/sws_labels_test-gold.csv','a') as lg_f:
            csvwriter = csv.writer(lg_f, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow([lg])

    test={}
    # keep track of words we have already see in the example
    temp = []
    for word in words:
        if word in temp:
            # if this word has already been see in this example, do nothing
            pass
        else:
            # if this the first time that we have seen this word, update the dictionary
            temp.append(word)
            if word in vocab:
                number = np.where(np.array(vocab)==word)[0][0]+1
                test.update({number:1})
    test_examples.append(test)

# pickle the test example dictionaries
with open(os.getcwd()+'/processed_test/sws_test-gold.pkl', 'wb') as fp:
    pickle.dump(dds, fp)


