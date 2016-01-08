# Similar Language Identification

A machine learning system for classifying texts written in similar languages.

Term Project, CS229 Machine Learning at Stanford University in Fall 2015

## Overview

This is a machine learning system that demonstrates a novel hierarchical method to classify sentences written in very similar languages or variations of languages. It was done as a term project for CS229 Machine Learning at Stanford University in Fall 2015. The repository contains training and testing data we used to develop the system, code, and a paper presenting our findings. The hierarchical nature of our system renders it robustness and scalability as well as competitive accuracy of classification.

We demonstrate our method by contributing to the DSL-2015 Shared Task. More information about the task can be found in this github [repo](https://github.com/Simdiva/DSL-Task/tree/master/data/DSLCC-v2.0/). The Task provide a training corpus consisting of sentences from journalistic articles written in 13 languages. The 13 languages are divided into six language groups, within each the languages are very similar and difficult to distinguish from one another. The table below lists the 13 languages and their division into groups. Test datasets with and without labels are also provided by the task. More details on the data sets can be found in the DSL-Task github [repo](https://github.com/Simdiva/DSL-Task/blob/master/data/DSLCC-v2.0/train-dev/README.md). 

| Langugae group             | Languages                                               |
| -------------------------- | ------------------------------------------------------- |
| South-Eastern Slavic (ses) | Bulgarian (bg), Macedonian (mk)                         |
| South-Western Slavic (sws) | Bosnian (bs), Croatian (hr), Serbian (sr)               |
| West Slavic (ws)           | Czech (cz), Slovak (sk)                                 |
| Spanish (es)               | Argentine Spanish (esar), Peninsular Spanish (eses)     |
| Portuguese (pt)            | Brazilian Portuguese (ptbr), European Portuguese (ptpt) |
| Austronesian (aus)         | Indonesian (id), Malay (my)                             |


Our approach for classification is a two-level hierarchy. We first identify if a sentence belongs to a language group, which is trivial and can be done with a simple method; within each language group, we then train a support vector machine or ensembles of SVMs to pinpoint which language to label the sentence (see schematic below). This method is easily scalable to include tens and hundreds of languages and incorporate more training data. At the langugae group level, training the classifier involves word-counting and identifying most common words within languages; this can be scaled up using a cluster file system and a MapReduce program. Within each language group, there are typically only 2 to 3 languages; this also makes parallelization easy and the number of features used in SVMs well-controlled.

![schematic](https://github.com/shenglan0407/Similar_language_identification/blob/master/writing/schema.jpeg)

## Performance
We tested our method with 1000 test examples per language, using [this dataset](https://github.com/Simdiva/DSL-Task/blob/master/data/DSLCC-v2.0/gold/test-gold.txt) from the DSL-Task. The confusion matrix summarizes our results.
![confusion matrix](https://github.com/shenglan0407/Similar_language_identification/blob/master/writing/Final_confusion_matrix.png)

Group accuracy represents the fraction of test examples correctly classified into their respective language groups during the first step of the hierarchical method. The upshot is using only the 1000 most frequently used words, a small fraction of all available vocabulary in each language, we can achieve near perfect accuracy distinguishing between language groups (we choose to combine the South-western Slavic and the West Slavic groups into one group as that increases our final accuracy). The final accuracy indicates the fraction of test examples correctly labeled by language after the second step of our methods involving ensemble SVMs. After impelmenting a feature selection procedure with a tfidf-type ranking index, the number of features for these SVMs within language groups are in the order of tens of thousands, which are very manageable and do not require extraordinary computation resources. In short, in designing our method we try to make it efficient and scalable. Details on the method are in our [paper](https://github.com/shenglan0407/Similar_language_identification/blob/master/writing/final.pdf)


## Organization
Write about organization of this repo

## Usage
How to run the code: currently a demo and not meant for general usage

## Contributing
1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request

## Credit
Daniel Levy and Shenglan Qiao (the Authors) contributed equally to this project.

Copyright 2015 Stanford University and the Authors
