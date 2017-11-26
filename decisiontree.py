# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:57:23 2017

Casey Astiz, Ben Brown, & Chloe Johnson
Professor Scharstein
CSCI 0451
December 8, 2017

"""

import csv
import numpy as np

#globals
data_file = 'Tweets.csv'

def main():
    data, labels = readfile(data_file)

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
        print (labels[:5])
    return (data,labels)

def extract_features(row):
    """takes a row of data as a dictionary mapping column header to column value
    returns label as an integer (negative -> -1, neutral -> 0, positive -> 1)
    returns data as a numpy array of floats
    """
    label = 1 if row['airline_sentiment'] == 'positive' else \
            0 if row['airline_sentiment'] == 'neutral' else -1

    return np.zeros(1), label

if __name__ == "__main__":
    main()
