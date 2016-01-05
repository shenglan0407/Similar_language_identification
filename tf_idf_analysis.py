import numpy as np
import time
import os
import pickle
import matplotlib.pyplot as plt
import operator

import csv
import sys
import getopt
import gc

sys.path.insert(0, '/Users/shenglanqiao/Documents/Classes/CS229/liblinear-2.1/python')
sys.path.insert(0, '/srv/zfs01/user_data/shenglan/liblinear-2.1/python')

import liblinear

from liblinearutil import *
languages = ['cz','sk','bs','hr','sr']
associationTable = {'cz': 0, 'sk' : 1,'bs': 2, 'hr': 3, 'sr': 4}

def getNgramsWordTfDfExtractor(p):
    def getNgrams(s,doc_count):
        s = s+' #end#'
        
        if p > 1:
            s = '#begin# '+s
        n = len(s)
        
        featureVector = {}
        
        words = s.split(' ')
        temp = []
        for i in range(len(words)-1):
            ngram = ' '.join(words[i:i+p])
            
            if ngram not in associationNgrams:
                associationNgrams[ngram] = count[0]
                count[0] += 1

                associationTf.update({ngram:1.})
                associationDf.update({ngram:1.})
                
            else:
                associationTf[ngram]+=1.
                if ngram not in temp:
                    associationDf[ngram]+=1.
            temp.append(ngram)
    return getNgrams

def getNgramsCharacterTfDfExtractor(p):
    def getNgrams(s,doc_count):
        s = ' '*(p-1)+s+' '*p
        n = len(s)

        featureVector = {}
        temp = []
        for i in range(n-p):
            ngram = s[i:i+p]
            if ngram not in associationNgrams:
                associationNgrams[ngram] = count[0]
                count[0] += 1
                
                associationTf.update({ngram:1.})
                associationDf.update({ngram:1.})
            else:

                associationTf[ngram]+=1.
                if ngram not in temp:
                    associationDf[ngram]+=1.
            temp.append(ngram)
    return getNgrams
    
def fileToTfDf(f, P=5):
    start_time = time.time()
    doc_count = 0
    
    for line in f:
        doc_count+=1
        sentence, label = line.split('\t')

        features = {}
        if char_gram:
            getNgramsCharacterTfDfExtractor(P)(sentence,doc_count)
        else:
            getNgramsWordTfDfExtractor(P)(sentence,doc_count)

    f.close()
    print 'Loading took ', time.time() - start_time

def getNgramsWordFeatureExtractor(p):
    def getNgrams(s):
        s = s+' #end#'
        
        if p > 1:
            s = '#begin# '+s
        n = len(s)
        
        featureVector = {}
        
        words = s.split(' ')
        
        for i in range(len(words)-1):
            ngram = ' '.join(words[i:i+p])
            if ngram in r_associationNgrams:
                if ngram not in featureVector:
                    featureVector.update({r_associationNgrams[ngram]:1})
        return featureVector
    return getNgrams

def getNgramsCharacterFeatureExtractor(p):
    def getNgrams(s):
        s = ' '*(p-1)+s+' '*p
        n = len(s)

        featureVector = {}
        for i in range(n-p):
            ngram = s[i:i+p]
            if ngram in r_associationNgrams:
                if ngram not in featureVector:
                    featureVector.update({r_associationNgrams[ngram]:1})

        return featureVector
    return getNgrams

def fileToFeatureVector(f, P=5):
    start_time = time.time()
    X = []
    y = []
    for line in f:
        sentence, label = line.split('\t')

        features = {}

        #for p in range(1,P+1):
        #    
            
        if char_gram:
            features.update(getNgramsCharacterFeatureExtractor(P)(sentence))
        else:
            features.update(getNgramsWordFeatureExtractor(P)(sentence))
            
        X.append(features)
        y.append(associationTable[label.rstrip()])

    f.close()
    print 'Loading took ', time.time() - start_time
    return X,y
    
def AccuracyByFeatureNum():
    f_test = open(os.getcwd()+'/devel_data/ws_sws_devel.txt','r')
    f_train = open(os.getcwd()+'/partial_train_data/90k_sws_ws_train.txt','r')


    X_train,y_train = fileToFeatureVector(f_train, P = Ngrams)
    X_test, y_test = fileToFeatureVector(f_test, P = Ngrams)
    
    param = parameter('-c 0.01')
    p = problem(y_train, X_train)
    m = train(p,param)
    print 'test accuracy:'
    prediction, testing_accuracy, _ = predict(y_test, X_test, m)
    print 'training accuray:'
    _,train_accuracy, _ = predict(y_train, X_train, m)

    train_accuracies.append(train_accuracy)
    test_accuracies.append(testing_accuracy)
    preds.append(prediction)
    if len(test_labels) == 0:
        test_labels.extend(y_test)
    
    f_train.close()
    f_test.close()
    
def usage():
    print 'tf_idf_analysis.py -c <CharOrWord> -n <NumGrams> -t <TopGrams>'



if __name__ == "__main__":
    
    argv = sys.argv[1:]
    char_gram = True
    Ngrams = 1
    top_gram = 100
    
    try:
        opts, args = getopt.getopt(argv,"hc:n:t:",["CharOrWord","NumGrams","TopGrams"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-c","--CharOrWord"):
            if int(arg) == 1:
                char_gram = True
                
            elif int(arg) == 2:
                char_gram = False
            else:
                print "option does not exist. 1 for character ngrams and 2 for word ngrams"
                sys.exit(2)
        elif opt in ("-n","--NumGrams"):
            Ngrams = int(arg)
        elif opt in ("-t","--TopGrams"):
            top_grams = int(arg)
    
    
    if char_gram:
        print "Running SVMs for character %d-grams" % Ngrams
    else:
        print "Running SVMs for word %d-grams" % Ngrams
    
    associationNgrams = {}
    associationTf = {}
    associationDf = {}

    count = [0]
    
    f_train = open(os.getcwd()+'/partial_train_data/90k_sws_ws_train.txt','r')
    n_doc = sum([1 for line in f_train])
    print 'there are %d documents' % n_doc
    f_train.close()

    f_train = open(os.getcwd()+'/partial_train_data/90k_sws_ws_train.txt','r')

    fileToTfDf(f_train,P = Ngrams)

    associationTfIdf = {}
    for ngram in associationDf.keys():
        tf_idf = associationTf[ngram]*np.log(n_doc/associationDf[ngram])
        associationTfIdf.update({ngram:tf_idf})
    
    sorted_tfidf = sorted(associationTfIdf.items(), key=operator.itemgetter(1))

    train_accuracies = []
    test_accuracies = []
    test_labels = []
    preds = [] 
    
    n_grams_to_use = [400,800,1200,1600,2000,\
                    5000,10000,20000,30000,40000,50000,60000,70000,100000]
    for item in n_grams_to_use:
        r_associationNgrams = {k: associationNgrams[k] for k in \
         np.array(sorted_tfidf[-item:])[:,0]}
        AccuracyByFeatureNum()
        gc.collect()
    
    results_to_save = {'tr_ac' : np.array(train_accuracies)
    ,'te_ac' : np.array(test_accuracies)
    , 'test_labels' : test_labels
    , 'predictions' : preds
    , 'features_sizes' : n_grams_to_use
    }
    
    if char_gram:
        pickle.dump(results_to_save,open(os.getcwd()+('/test_results/sws_ws_char_%d-gram_tfidf.pkl' % Ngrams),'wb'))
    else:
        pickle.dump(results_to_save,open(os.getcwd()+('/test_results/sws_ws_word_%d-gram_tfidf.pkl' % Ngrams),'wb'))
   

