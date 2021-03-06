# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:57:23 2017

Casey Astiz, Ben Brown, & Chloe Johnson
Professor Scharstein
CSCI 0451
December 8, 2017

"""

import csv, os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from decisiontree import DecisionTree
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.svm import SVC

#globals
data_file = 'Tweets.csv'
test_size = 0.3
word_file = 'words.txt'

def main():

    word_list = create_words()

    data, labels = readfile(word_list)
    tree = DecisionTree() 
    data_train, data_test, labels_train, labels_test = \
        train_test_split(data, labels, test_size=test_size, random_state=42)

    ## calls our tree algorithm and prediction method ##
    tree.train(data_train, labels_train)
    labels_pred = tree.predict(data_test)
    compute_accuracy(labels_test, labels_pred)


    ## Looking at other algorithm's performance on the dataset ##

#    tree2 = DecisionTreeClassifier(max_depth = 10, min_samples_leaf = 5)
#    # tree2 = DecisionTreeClassifier() #not providing any params
#    tree2.fit(data_train, labels_train)
#    labels_pred2 = tree2.predict(data_test)
#    print(list(set(labels_pred2)))

#    logreg = linear_model.LogisticRegression(C=1e5)
#    logreg.fit(data_train, labels_train)
#    labels_pred3 = logreg.predict(data_test)
#    
#    clf = SVC()
#    svmmodel = clf.fit(data_train, labels_train)
#    labels_pred4 = svmmodel.predict(data_test)
    
    
#    compute_accuracy(labels_test,labels_pred2)
#    compute_accuracy(labels_test,labels_pred3)
#    compute_accuracy(labels_test,labels_pred4)

    # Random Forest ##
#    clf2 = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0, min_samples_leaf=5)
#    clf2.fit(data_train, labels_train)
#    labels_pred5 = clf2.predict(data_test)
#    compute_accuracy(labels_test, labels_pred5)

def create_words():
    """if it exists, opens wordfile. Otherwise, creates words and writes to wordfile """
    if (os.path.isfile(word_file)):
        return open(word_file).read().strip().split('\n')
    else:
        positive = {}
        neutral = {}
        negative = {}

        with open(data_file) as tweets:
            reader = csv.DictReader(tweets)
            for row in reader:
                for word in row['text'].strip().split():
                    word = word.lower()
                    word = "".join(c for c in word if c not in ('!','.',',', '?','"', "'")) # https://stackoverflow.com/questions/16050952/how-to-remove-all-the-punctuation-in-a-string-python
                    label = row['airline_sentiment']

                    # Positive labels
                    if word in positive.keys():
                        if (label == 'positive'):
                            positive[word] += 1
                        else:
                            pass
                    else:
                        if (label == 'positive'):
                            positive[word] = 1
                        else:
                            positive[word] = 0

                    # Neutral labels
                    if word in neutral.keys():
                        if (label == 'neutral'):
                            neutral[word] += 1
                        else:
                            pass
                    else:
                        if (label == 'neutral'):
                            neutral[word] = 1
                        else:
                            neutral[word] = 0

                    # Negative labels
                    if word in negative.keys():
                        if (label == 'negative'):
                            negative[word] += 1
                        else:
                            pass
                    else:
                        if (label == 'negative'):
                            negative[word] = 1
                        else:
                            negative[word] = 0

    positive_list = []
    neutral_list = []
    negative_list = []

    positive_multiplier = 3
    neutral_multiplier = 3
    negative_multiplier = 3

    min_occurrences = 20

    for word in positive:
        if ((positive[word] >= positive_multiplier * negative[word]) and (positive[word] >= positive_multiplier * neutral[word])) and ((negative[word] != 0 and neutral[word] != 0) or positive[word] > min_occurrences):
            positive_list.append(word)
    for word in neutral:
        if ((neutral[word] >= neutral_multiplier * negative[word]) and (neutral[word] >= neutral_multiplier * positive[word])) and ((negative[word] != 0 and positive[word] != 0) or neutral[word] > min_occurrences):
            neutral_list.append(word)
    for word in negative:
        if ((negative[word] >= negative_multiplier * positive[word]) and (negative[word] >= negative_multiplier * neutral[word])) and ((neutral[word] != 0 and positive[word] != 0) or negative[word] > min_occurrences):
            negative_list.append(word)

    word_list = positive_list + neutral_list + negative_list
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
    """extracts the airline from a data row"""
    airlines = ['Virgin America', 'United', 'Southwest', 
                'Delta', 'US Airways', 'American']
    x = np.zeros(len(airlines))
    try:
        index = airlines.index(row['airline'])
        x[index] = 1
    except ValueError:
        pass
    return x

def compute_accuracy(labels_test, labels_pred):
    """computes the accuracy"""
    acc = sum([int(x == y) for x, y in zip(labels_test, labels_pred)]) / float(len(labels_test))
    print ("Accuracy: {}".format(acc))

if __name__ == "__main__":
    main()
