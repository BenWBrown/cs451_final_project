# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:57:23 2017

Casey Astiz, Ben Brown, & Chloe Johnson
Professor Scharstein
CSCI 0451
December 8, 2017

code that imports our decision tree, trains it on a dataset, then tests on a validation set
"""

import csv
import numpy as np
from sklearn.model_selection import train_test_split
from decisiontree import DecisionTree

#globals
data_file = 'Tweets.csv'
test_size = 0.3

def main():
    data, labels = readfile(data_file)
    tree = DecisionTree()
    data_train, data_test, labels_train, labels_test = \
        train_test_split(data, labels, test_size=test_size, random_state=42)

    tree.train(data_train, labels_train)
    labels_pred = tree.predict(data_test)
    compute_metrics(labels_test, labels_pred)

def readfile(data_file):
    """Read in Kaggle dataset (Tweets.csv) and output tuple containing
    an array of features and the data labels"""
    with open(data_file) as tweets:
        reader = csv.DictReader(tweets)
        data = []
        labels = []
        for row in reader:
            x, y = extract_features(row)
            data.append(x)
            labels.append(y)
    return (data,labels)

def extract_features(row):
    """takes a row and extracts the data and the label"""
    label = extract_label(row)
    text = extract_text(row)
    data = vectorize(text)

    return data, label

def extract_label(row):
    """extracts a label from a data row.
    if the dataset is changed, this needs to be refactored"""
    return 1 if row['airline_sentiment'] == 'positive' else \
           0 if row['airline_sentiment'] == 'neutral' else -1

def extract_text(row):
    """extracts the tweet text from row data.
    if the dataset is changed, this needs to be refactored"""
    return row['text']

def vectorize(text):
    """vectorizes tweet text into a numpy array.
    This function is independent of the dataset"""
    #TODO
    return np.zeros(1)

def compute_metrics(labels_test, labels_pred):
    acc = sum([int(x == y) for x, y in zip(labels_test, labels_pred)]) / float(len(labels_test))
    print ("Accuracy: {}".format(acc))

if __name__ == "__main__":
    main()
