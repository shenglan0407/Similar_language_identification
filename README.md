# Similar Language Identification

A machine learning system for classifying texts written in similar languages.

Term Project, CS229 Machine Learning at Stanford University in Fall 2015

## Overview

This is a machine learning system that demonstrates a novel hierarchical method to classify sentences written in very similar languages or variations of languages. It was done as a term project for CS229 Machine Learning at Stanford University in Fall 2015. The repository contains training and testing data we used to develop the system, code, and a paper presenting our findings. The hierarchical nature of our system renders it robustness and scalability as well as competitive accuracy of classification.

We demonstrate our method by contributing to the DSL-2015 Shared Task. More information about the task can be found in this github repo (https://github.com/Simdiva/DSL-Task/tree/master/data/DSLCC-v2.0/). The Task provide a training corpus consisting of sentences from journalistic articles written in 13 languages. The 13 languages are divided into six language groups, within which the languages are very similar and difficult to distinguish from one another. Test datasets with and without labels are also provided by the task. Details on the data sets can be found in the DSL-Task github repo. 

Our approach for classification is a two-level hierarchy. We first identify if a sentence belongs to a language group, which is trivial and can be done with a simple method. Within each language group, we then train a support vector machine or ensembles of SVMs to pinpoint which language to label the sentence. This method is easily scalable to include tens and hundreds of languages and incorporate more training data. At the langugae group level, training the classifier involves word-counting and identifying most common words within languages; this can be scaled up using a cluster file system and a MapReduce program. Within each language group, there are typically only 2 to 3 languages; this also makes parallelization easy and the number of features used in SVMs well-controlled.

## Organization
Write about organization of this repo

## Usage
How to use the code

## Credit
Daniel Levy and Shenglan Qiao (the Authors) contributed equally to this project.
Copyright 2015 Stanford University and the Authors
