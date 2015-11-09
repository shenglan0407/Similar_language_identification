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

csvfile=open(os.getcwd()+'/data/top_words_train.csv')
reader = csv.DictReader(csvfile)
n_lgs = 13
lgs = []
word_list = []
freq_list = []

for ii in range(n_lgs):
    lg = reader.next()[None][0]
    words = [cell.decode('utf-8') for cell in reader.next()[None]]
    freqs = [cell for cell in reader.next()[None]]
    words = words[0].split()
    freqs = freqs[0].split()
    freqs = [float(item) for item in freqs]
    lgs.append(lg)
    word_list.append(words)
    freq_list.append(freqs)
    #print lg, words,freqs
lgs = np.array(lgs)
csvfile.close()

test_set = open(os.getcwd()+'/data/data_no_punc_no_digit_lowercase_test-gold.txt','r')
confusion_matrix = np.zeros((n_lgs,n_lgs))

for line in test_set:
    words = line.split()
    lg_label = words[-1]
    # not testing examples labeled xx 
    if  lg_label == 'xx':
        pass
    else:
        words=words[0:-1]
        words = [wd.decode('utf-8') for wd in words]
        lg_scores = []
        for lg_id in range(len(lgs)):
            top_words = word_list[lg_id][0:998]
            line_vector = np.zeros(len(top_words))
            lg_vector = freq_list[lg_id][0:998]

            for wd_id in range(len(line_vector)):
                if word_list[lg_id][wd_id] in words:
                    line_vector[wd_id] = 1
            this_lg_score = np.dot(line_vector,lg_vector)
            lg_scores.append(this_lg_score)
        predict = np.argmax(lg_scores)
        confusion_matrix[np.where(lgs==lg_label)[0][0]][predict] +=1   
        
        
with open(os.getcwd()+'/test_results/test-gold_ConfMat.csv', 'wb') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(lgs)
    for row in confusion_matrix:
        csvwriter.writerow(row)

test_set.close()
        