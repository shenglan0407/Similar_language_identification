##############################################################################
# Copyright 2015 Stanford University and the Author
#
# Author: Shenglan Qiao
# 
# This script take the training corpus and removes punctuations, digits, and turns
# letters into lowercase.
#
#############################################################################


##############################################################################
# Imports
##############################################################################

import os
import string

##############################################################################
# Code
##############################################################################

# input_f = open('/Users/shenglanqiao/Documents/GitHub/DSL-Task/data/DSLCC-v2.0/gold/test-gold.txt','r')
output_f = open(os.getcwd()+'/data/data_no_punc_no_digit_train.txt','r+')

#create the .txt with digits and punctuations removed but keep the upper/lowercases of the original corpus 
# for line in input_f:
#     out_line = line.translate(string.maketrans("",""),string.digits+string.punctuation)
#     output_f.write(out_line)
# 
# input_f.close()

# create the .txt file with only lowercase letters
this_file = output_f
output_f = open(os.getcwd()+'/data/data_no_punc_no_digit_lowercase_train.txt','w')
for line in this_file:
    out_line = line.decode('utf-8').lower()
    output_f.write(out_line.encode('utf-8'))

this_file.close()
output_f.close()
