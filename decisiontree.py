import numpy as np
from tqdm import tqdm
import random

class DecisionTree:

    max_depth = 10
    min_terminal_size = 5
    k = 100

    def __init__(self, **kwargs):
        self.root = None
        self.label_list = [-1, 0, 1]
        try:
            self.max_depth = kwargs['max_depth']
        except:
            pass

    def train(self, data, labels):
        """trains the decision tree on a given dataset"""
        self.label_list = list(set(labels))
        combined_data = list(zip(data, labels))
        self.root = self.build_tree(combined_data, self.max_depth)

    def predict(self, data):
        """predicts labels for given data"""
        predictions = []
        for vector in data:
            predictions.append(self.predict_vector(self.root, vector))
        return predictions

    def predict_vector(self, node, vector):
        """predicts label for given vector"""
        if node["terminal"]:
            return node["label"]
        if vector[node["attr"]] <= node["value"]:
            return self.predict_vector(node["left"], vector)
        else:
            return self.predict_vector(node["right"], vector)


    def gini(self, segment):
        """computes the Gini impurity"""
        segment_labels = []
        for element in segment:
            segment_labels.append(element[1])

        s = 0
        length = float(len(segment_labels))

        for label in self.label_list:
            if len(segment_labels)!= 0:
                p_i = 0
                for x in segment_labels:
                    p_i += int(label == x)

                p_i /= length

                s += p_i ** 2
        return 1 - s

    def build_tree(self, combined_data, max_depth):
        """builds the decision tree"""
        if max_depth <= 0:
            return self.terminal(combined_data)
        split = self.get_split(combined_data)
        if len(split["segments"][0]) < self.min_terminal_size or len(split["segments"][1]) < self.min_terminal_size:
            return self.terminal(combined_data)
        left = self.build_tree(split["segments"][0], max_depth-1)
        right = self.build_tree(split["segments"][1], max_depth-1)
        return {"terminal": False,
                "label": None,
                "right": right,
                "left": left,
                "attr": split["attr"],
                "value": split["value"]}

    def most_frequent_label(self, combined_data):
        """select the most common label (positive, neutral, or negative) from
        a segment of data assigned to terminal nodes"""
        labels = [x[1] for x in combined_data]
        return max(set(labels), key=labels.count)

    def split(self, attr, value, data):
        """takes in an attribute and a value to split the data"""
        left = list()
        right = list()

        for entry in data:
            if entry[0][attr] <= value:
                left.append(entry)
            else:
                right.append(entry)
        return left, right


    def get_split(self, data):
        """finds the best split point for the dataset"""
        best_split = 1000
        best_value = 1000
        best_gini = 1000
        best_segments = None

        for feat_index in tqdm(range(len(data[0][0]) - 1)):
            if (len(data) > self.k):
                sample = random.sample(data, self.k)
            else:
                sample = data

            for entry in sample:
                segments = self.split(feat_index, entry[0][feat_index], data)
                temp_gini = 0
                total_size = sum([len(segment) for segment in segments])
                for segment in segments:
                    temp_gini += self.gini(segment) * (len(segment) / float(total_size))
                if temp_gini < best_gini:
                    best_split = feat_index
                    best_value = entry[0][feat_index]
                    best_gini = temp_gini
                    best_segments = segments
        return {'attr' : best_split, 'value' : best_value, 'segments' : best_segments}

    def terminal(self, data):
        """creates a terminal node"""
        return {"terminal": True,
                "label": self.most_frequent_label(data),
                "right": None,
                "left": None,
                "attr": 0,
                "value": 0}



if __name__ == "__main__":
    tree = DecisionTree()
    test_data = [ (np.array([1, 2]), 0) ,(np.array([0, 2]), 0) ]
    split = tree.get_split(test_data)
