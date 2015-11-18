import os
import numpy as np
import heapq
import csv

lgs_sws = ['bs','hr','sr']
f=open(os.getcwd()+'/data/data_no_punc_no_digit_lowercase_train.txt','r')
with open(os.getcwd()+'/partial_train_data/18k_sws_train.txt','w') as output:
    for ii in range(252000):
        line = f.readline()
        lg = line.split()[-1]
        if ii%3==0 and lg in lgs_sws:
            output.write(line)
f.close()