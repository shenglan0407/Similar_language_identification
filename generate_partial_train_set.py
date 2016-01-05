import os
import numpy as np
import heapq
import csv

lgs_ws_sws = ['cz','sk','bs','hr','sr']
lgs_ws = ['cz','sk']
lgs_sws = ['bs','hr','sr']
lgs_ses = ['bg','mk']
lgs_sp = ['esar','eses']
lgs_pt = ['ptbr','ptpt']
lgs_aus=['id','my']

f=open(os.getcwd()+'/devel_data/data_no_punc_no_digit_lowercase_dev.txt','r')
with open(os.getcwd()+'/devel_data/ws_sws_devel.txt','w') as output:
    for ii in range(28000):
        line = f.readline()
        lg = line.split()[-1]
        if ii%1==0 and lg in lgs_ws_sws:
            output.write(line)
f.close()