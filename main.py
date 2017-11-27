# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:57:23 2017

Casey Astiz, Ben Brown, & Chloe Johnson
Professor Scharstein
CSCI 0451
December 8, 2017

code that imports our decision tree, trains it on a dataset, then tests on a validation set
"""

import csv, os
import numpy as np
from sklearn.model_selection import train_test_split
from decisiontree import DecisionTree

#globals
data_file = 'Tweets.csv'
test_size = 0.3
word_file = 'words.txt'

def main():
    word_list = create_words()

    data, labels = readfile(word_list)
    tree = DecisionTree() #todo: some sort of hyper-params
    data_train, data_test, labels_train, labels_test = \
        train_test_split(data, labels, test_size=test_size, random_state=42)

    tree.train(data_train, labels_train)
    labels_pred = tree.predict(data_test)
    compute_metrics(labels_test, labels_pred)

def create_words(num_words=200):
    """if it exists, opens wordfile. Otherwise, creates words and writes to wordfile """
    if (os.path.isfile(word_file)):
        return open(word_file).read().strip().split('\n')

    words = {}
    with open(data_file) as tweets:
        reader = csv.DictReader(tweets)
        for row in reader:
            for word in row['text'].strip().split():
                try:
                    words[word] += 1
                except:
                    words[word] = 1

    word_list = [word[0] for word in sorted(words.items(), key=lambda x: x[1], reverse=True)][:num_words]
    # print ([word for word in sorted(words.items(), key=lambda x: x[1], reverse=True)][:100])
    #sort list
    with open(word_file, 'w') as f:
        f.writelines(["%s\n" % item  for item in word_list])
    return word_list


def readfile(word_list):
    """Read in Kaggle dataset (Tweets.csv) and output tuple containing
    an array of features and the data labels"""
    with open(data_file) as tweets:
        reader = csv.DictReader(tweets)
        data = []
        labels = []
        for row in reader:
            x, y = extract_features(row, word_list)
            data.append(x)
            labels.append(y)
        print (data[0])
    return (data,labels)

def extract_features(row, word_list):
    """takes a row and extracts the data and the label"""
    label = extract_label(row)
    text = extract_text(row)
    data = vectorize(text, word_list)

    airline_data = extract_airline(row)
    data = np.append(airline_data, data)

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

def vectorize(text, word_list):
    """vectorizes tweet text into a numpy array.
    This function is independent of the dataset"""
    return np.array( [float(word in text.split()) for word in word_list] )


def extract_airline(row):
    airlines = ['Virgin America', 'United', 'Southwest', 'Delta', 'US Airways', 'American']
    x = np.zeros(len(airlines))
    try:
        index = airlines.index(row['airline'])
        x[index] = 1
    except ValueError:
        pass
    return x

def compute_metrics(labels_test, labels_pred):
    acc = sum([int(x == y) for x, y in zip(labels_test, labels_pred)]) / float(len(labels_test))
    print ("Accuracy: {}".format(acc))

if __name__ == "__main__":
    main()
