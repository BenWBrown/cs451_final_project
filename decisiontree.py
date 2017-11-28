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

    def predict(self, data):
        """predicts labels for given data"""
        #TODO
        return [1] * len(data)

    def gini(self, labels):
        s = 0
        for label in self.label_list:
            p_i = sum([int(label == x) for x in labels ]) / float(len(labels))
            s += p_i ** 2
        return 1 - s


if __name__ == "__main__":
    tree = DecisionTree()
    test_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, -1])
    print (tree.gini(test_labels))
