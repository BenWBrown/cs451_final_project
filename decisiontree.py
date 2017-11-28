import numpy as np

class DecisionTree:

    max_depth = 4

    def __init__(self, **kwargs):
        #for all default hyperparams
        self.label_list = [-1, 0, 1] #TODO: GET RID OF THIS LOL
        try:
            self.max_depth = kwargs['max_depth']
        except:
            pass

    def train(self, data, labels):
        """trains the decision tree on a given dataset"""
        self.label_list = list(set(labels))
        combined_data = list(zip(data, labels))
        #print(self.gini(combined_data))
        #get_split(self, combined_data)

    def predict(self, data):
        """predicts labels for given data"""
        #TODO
        return [1] * len(data)

    def gini(self, segment):

        segment_labels = []
        for element in segment:
            segment_labels.append(element[1])

        s = 0
        for label in self.label_list:
            if len(segment_labels)!= 0:
                p_i = sum([int(label == x) for x in segment_labels ]) / float(len(segment_labels))
                s += p_i ** 2
        return 1 - s

    def split(self, attr, value, data):
        """takes in an attribute and a value to split the data"""
        #Should we be making 3 splits or 2?
        left, center, right = list(), list(), list()

        for entry in data:
            if entry[attr] < value:
                left.append(entry)
            elif entry[attr] == value:
                center.append(entry)
            else:
                right.append(entry)
        return left, center, right

    def get_split(self, data):
        """finds the best split point for the dataset"""
        best_split = 1000
        best_value = 1000
        best_gini = 1000
        best_segments = None

        for i in range(len(self.label_list) - 1):
            for entry in data:
                segments = self.split(i, entry[i], data)
                temp_gini = 0
                total_size = sum([len(segment) for segment in segments])
                for segment in segments:
                    temp_gini += (1.0 - self.gini(segment)) * (len(segment) / total_size)
                if temp_gini < best_gini:
                    best_split = i
                    best_value = entry[i]
                    best_gini = temp_gini
                    best_segments = segments
        return {'split' : best_split, 'value' : best_value, 'segments' : best_segments}


if __name__ == "__main__":
    tree = DecisionTree()
    test_labels = np.array([(123, 1),(235423, 1),(1231, 0),(12341,0)])
    print (tree.gini(test_labels))
