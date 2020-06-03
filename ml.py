# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 16:15:34 2020

@author: tomma
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

path_dataset = '/Users/tomma/Documents/IE HST/3rd term/MLOPS/IA/creditcard.csv'
dataset = pd.read_csv(path_dataset)
len(dataset)
train = dataset[0:284000]
test = dataset[284001:284807]
test_copy = test.Class
test = test.drop('Class', 1)

target = 'Class'

def split_dataset(df,random_seed, size = 0.2):
    y = df[target]
    X = df.drop(target,axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state = random_seed)
    return X_train, X_test, y_train, y_test

def train_and_evaluate_classifier(X, yt, estimator, grid):
    """Train and Evaluate a estimator (defined as input parameter) on the given labeled data using accuracy."""
    
    # Cross validation
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    classifier = GridSearchCV(estimator=estimator, cv=cv,  param_grid=grid, error_score=0.0, n_jobs = -1, verbose = 5, scoring='f1')
    
    # Train the model over and tune the parameters
    print("Training model")
    classifier.fit(X, yt)

    # CV-score
    print("CV-scores for each grid configuration")
    means = classifier.cv_results_['mean_test_score']
    stds = classifier.cv_results_['std_test_score']
    for mean, std, params in sorted(zip(means, stds, classifier.cv_results_['params']), key=lambda x: -x[0]):
        print("Accuracy: %0.3f (+/-%0.03f) for params: %r" % (mean, std * 2, params))
    print()

    return classifier

'''

POSSIBLE FEATURE ENGINEERING AND DATA CLEANING
Add a simple feature to your code. (e.g. add regularization to your model, normalize the data, etc)
'''

X_train, X_test, y_train, y_test = split_dataset(train,100)
train_and_evaluate_classifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
model.score(X,y)
print("Accuracy = {0:.4f}".format(accuracy_score(y_test,predictions)))




