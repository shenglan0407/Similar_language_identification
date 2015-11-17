##############################################################################
# Copyright 2015 Stanford University and the Author
#
# Author: Shenglan Qiao
# 
# svm with LIBLINEAR, word one-gram
#
#############################################################################


##############################################################################
# Imports
##############################################################################
import pandas as pd
import numpy as np
import time
import os
import pickle
import csv
##############################################################################
# Code
##############################################################################

train_data = pickle.load(open(os.getcwd()+'/processed_training/sws_5400.pkl', 'wb'))