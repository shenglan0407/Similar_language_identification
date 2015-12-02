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
# Parameters
##############################################################################

# if true, will write the label to the test set to a .csv file
# only need to make true if it's the first test processing the test set
write_labels = False

# if ture, process the test set assuming bigrams are features
is_bigram = False

# path to where the vocab for this classifier is saved
vocab_file = os.getcwd()+'/processed_training/sws_train_5400_1gram_vocab_reduced.csv'

# where to save the processed test examples
# naming convention: LanguageGroup_model_numberOfExamples_TestSet.pkl.
# e.g.  sws_m2w_54k_test-gold.pkl means southwest slavic group, modle with word bigrams
# 54k examples and from test set 'test-gold'
save_path = os.getcwd()+'/processed_test/sws_m1wreduced_5400_test-gold.pkl'

##############################################################################
# Code
##############################################################################

# load the vocabulary learnt from training set
with open(vocab_file, 'r') as f:
    vocab = []
    csvreader =  csv.reader(f, delimiter=' ', quotechar='|')
    for row in csvreader:
        vocab.append(row[0])

# file that contains the test set
test_f=open(os.getcwd()+'/data/sws_test-gold.txt','r')

# language present in test set
all_lg = ['bs','hr','sr']

test_labels=[]
test_examples=[]
count = 0
for line in test_f:

    count +=1
    line=line.split()
    words=line[:-1]
    lg=line[-1]
    # record labels of of test set if write_labels is true
    if write_labels:
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
    # if is_gram is true, look for bigrams in test examples
    if is_bigram:
        bigrams = ['%s %s'%(words[ii],words[ii+1]) for ii in range(len(words)-1)]
        bigrams.append('#begin#' + words[0])
        bigrams.append( words[-1]+'#end#')
        
        for gram in bigrams:
            if gram in temp:
                # if this word has already been see in this example, do nothing
                pass
            else:
                # if this the first time that we have seen this word, update the dictionary
                temp.append(gram)
                if gram in vocab:
                    number = np.where(np.array(vocab)==gram)[0][0]+1
                    test.update({number:1})
    # if is_bigram is false, look for one-grams in test examples
    else:
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
with open(save_path, 'wb') as fp:
    pickle.dump(test_examples, fp)


