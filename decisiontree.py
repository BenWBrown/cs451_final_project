import numpy as np
from tqdm import tqdm
import random

class DecisionTree:

    max_depth = 4
    min_terminal_size = 5
    k = 100

    def __init__(self, **kwargs):
        self.root = None
        #for all default hyperparams
        #TODO: SET OTHER HYPERPARAMS
        self.label_list = [-1, 0, 1] #TODO: GET RID OF THIS LOL
        try:
            self.max_depth = kwargs['max_depth']
        except:
            pass

    def train(self, data, labels):
        """trains the decision tree on a given dataset"""
        self.label_list = list(set(labels))
        combined_data = list(zip(data, labels))
        self.root = self.build_tree(combined_data, self.max_depth)
        #print(self.gini(combined_data))
        #get_split(self, combined_data)

    def predict(self, data):
        """predicts labels for given data"""
        if self.root is None:
            return [0] * len(data)
        predictions = []
        for vector in data:
            predictions.append(self.predict_vector(self.root, vector))
        return predictions


    @staticmethod
    def predict_vector(node, vector):
        """predicts label for given vector"""
        if node["terminal"]:
            return node["label"]
        if vector[node["attr"]] < node["value"]:
            return self.predict_vector(node["left"])
        else:
            return self.predict_vector(node["right"])


    def gini(self, segment):
        """computes the Gini impurity"""
        segment_labels = []
        for element in segment:
            segment_labels.append(element[1])

        s = 0
        for label in self.label_list:
            if len(segment_labels)!= 0:
                p_i = sum([int(label == x) for x in segment_labels ]) / float(len(segment_labels))
                s += p_i ** 2
        return 1 - s

    def build_tree(self, combined_data, max_depth):
    """builds the decision tree"""        
        if max_depth <= 0:
            return self.terminal(combined_data)
        split = self.get_split(combined_data)
        print (split.keys())
        print (len(split['segments'][0]))
        print (len(split['segments'][1]))
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

    @staticmethod
    def most_frequent_label(combined_data):
        """select the most common label (positive, neutral, or negative) from
        a segment of data assigned to terminal nodes"""
        labels = [x[1] for x in combined_data]
        return max(set(labels), key=labels.count)

    def split(self, attr, value, data):
        """takes in an attribute and a value to split the data"""
        left, right = list(), list()

        for entry in data:
            # print(entry)
            # print(attr)
            # print(entry[0][attr])
            # print(value)
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
            for entry in random.sample(data, self.k):
                #split(attr, value, data)
                segments = self.split(feat_index, entry[0][feat_index], data)
                temp_gini = 0
                total_size = sum([len(segment) for segment in segments])
                for segment in segments:
                    temp_gini += (1.0 - self.gini(segment)) * (len(segment) / total_size)
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
    test_labels = np.array([(123, 1),(235423, 1),(1231, 0),(12341,0)])
    print (tree.gini(test_labels))
