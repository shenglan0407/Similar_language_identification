import csv
import os
import numpy as np

lgs = ['bs','hr','sr']
with open(os.getcwd()+'/data/sws_test-gold.txt','w') as output:
    with open(os.getcwd()+'/data/data_no_punc_no_digit_lowercase_test-gold.txt','r') as f:
        for line in f:
            lg = line.split()[-1]
            if lg in lgs:
                output.write(line)
                