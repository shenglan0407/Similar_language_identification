import os
import numpy as np
import heapq
import csv

lgs_ws = ['cz','sk']
lgs_sws = ['bs','hr','sr']
lgs_ses = ['bg','mk']
lgs_sp = ['esar','eses']
lgs_pt = ['ptbr','ptpt']
lgs_aus=['id','my']
f=open(os.getcwd()+'/data/data_no_punc_no_digit_lowercase_train.txt','r')
with open(os.getcwd()+'/partial_train_data/36k_aus_train.txt','w') as output:
    for ii in range(252000):
        line = f.readline()
        lg = line.split()[-1]
        if ii%1==0 and lg in lgs_aus:
            output.write(line)
f.close()