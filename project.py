from moteur_id3.id3 import ID3, NoeudDeDecision
from csv import DictReader
import pandas as pd
import numpy as np


class ResultValues:

    def __init__(self):
        # Do computations here
        # Task 1
        df = pd.read_csv('data/train_bin.csv')
        task1_train_data = parse_data(df)

        id3 = ID3()
        tree = id3.construit_arbre(task1_train_data)

        self.arbre = tree

        print('max height of the tree:', max_depth(tree))
        print('min height of the tree:', min_depth(tree))
        print('number of leaves in the tree:', get_leaf_count(tree))
        print('average height of the tree:', average_height(self.arbre))

        # Task 2
        df = pd.read_csv('data/test_public_bin.csv')
        task2_test_data = parse_data(df)
        task2_accuracy = test_stats(self.arbre, task2_test_data)
        print('accuracy is :', format(task2_accuracy, '.2%'))

        # Task 3

        self.faits_initiaux = initial_facts('data/train_bin.csv')
        self.regles = get_paths(self.arbre)

        # Task 4

        def tree_predict(t, data):

            pred_heart_disease = []
            for target, inp in data:
                if t.classifie(inp)[-1] == '1':
                    pred_heart_disease.append(inp)

            return pred_heart_disease

        print(tree_predict(self.arbre, task2_test_data))

        # Task 5
        self.arbre_advance = None

    def get_results(self):
        return [self.arbre, self.faits_initiaux, self.regles, self.arbre_advance]


# Static functions
def parse_data(dataframe):
    """ Parse dataframe into desired format.

       :param DataFrame dataframe: the dataframe to parse
       :return: a list of pairs
    """
    headers = list(dataframe)
    res = []
    for index, row in dataframe.iterrows():
        class_attr = {}
        for h in headers[:len(headers) - 1]:
            class_attr[h] = str(row[h])
        entry = [str(row[headers[-1]]), class_attr]
        res.append(entry)
    return res


def min_depth(t):
    """ Parse dataframe into desired format.

        :param NoeudDeDecision t: the tree to analyze
        :return: min height of the tree
    """
    if t.terminal():
        return 0
    else:
        children = t.enfants
        children_depth = []
        for e in children:
            children_depth.append(min_depth(children[e]))
        return min(children_depth) + 1


def get_leaf_count(t):
    """ Parse dataframe into desired format.

        :param NoeudDeDecision t: the tree to analyze
        :return: number of leaves
    """
    if t.terminal():
        return 1
    else:
        children = t.enfants
        children_leaves = []
        for e in children:
            children_leaves.append(get_leaf_count(children[e]))
        return sum(children_leaves)


def max_depth(t):
    """ Parse dataframe into desired format.

        :param NoeudDeDecision t: the tree to analyze
        :return: max height of the tree
    """
    if t.terminal():
        return 0
    else:
        children = t.enfants
        children_depth = []
        for e in children:
            children_depth.append(max_depth(children[e]))
        return max(children_depth) + 1

def average_height(t):
    """ Find the average height of a decision tree

        :param NoeudDeDecision t: the tree
        :return: average height of the tree
    """
    paths = get_paths(t)
    lengths = []
    for path in paths:
        lengths.append(len(path))

    ah = sum(lengths)/len(paths)

    return ah


def test_stats(tree, data):
    """ Test the tree on a test dataset.

        :param NoeudDeDecision tree: the tree to test
        :param list data: test data
        :return: accuracy of the classifications
    """
    print(data)
    success = 0
    for target, inp in data:
        if tree.classifie(inp)[-1] == target:
            success += 1

    return success / len(data)


def initial_facts(path):
    """ Retrieve initial facts from a csv file.

        :param str path: file path
        :return: initial facts
    """
    with open(path, 'r', encoding='utf-8-sig') as read_obj:
        # pass the file object to DictReader() to get the DictReader object
        csv_dict_reader = DictReader(read_obj)
        data = []
        # iterate over each line as a ordered dictionary
        for row in csv_dict_reader:
            # row variable is a dictionary that represents a row in csv
            data.append(list(list(e) for e in row.items()))

    return data

def get_paths(t):
    """ Get all the paths in a tree using DFS.

        :param NoeudDeDecision t: the tree to explore
        :return: list of paths in the tree
    """
    if t.terminal():
        return [[t.classe().upper()]]
    paths = []
    children = t.enfants
    for value, child in children.items():
        res_p = get_paths(child)
        for path in res_p:
            paths.append([[t.attribut, value]] + path)
    return paths

def explain_prediction(rules, datapoint):
    """ get prediction explanation for the datapoind based on the rules.

        :param list rules: the list of rules
        :param list datapoint: the datapoint attributes
        :return: the explanation of the prediction
    """
    for rule in rules:
        if all(i in datapoint for i in rule[:-1]):
            return print(rule[:-1], "=>", rule[-1])

    return "No explanation found. Ask a real doctor"