from csv import DictReader

import pandas as pd

from moteur_id3.id3 import ID3
from moteur_id3.noeud_de_decision import NoeudDeDecision
from moteur_id3_cts.id3_cts import ID3Cts


class ResultValues:

    def __init__(self):
        # Do computations here

        # Task 1

        df = pd.read_csv('data/train_bin.csv')
        task1_train_data = parse_data(df)

        id3 = ID3()
        tree = id3.construit_arbre(task1_train_data)

        self.arbre = tree

        print('***** Task 1 & 2 *****')
        print('max height of the tree:', max_depth(tree))
        print('min height of the tree:', min_depth(tree))
        print('average height of the tree:', "{:.2f}".format(average_height(self.arbre)))
        print('number of leaves in the tree:', get_leaf_count(tree))

        # Task 2
        df = pd.read_csv('data/test_public_bin.csv')
        task2_test_data = parse_data(df)
        task2_accuracy = test_stats(self.arbre, task2_test_data)
        print('accuracy is :', format(task2_accuracy, '.2%'))

        # Task 3

        self.faits_initiaux = initial_facts('data/train_bin.csv')
        self.regles = get_paths(self.arbre)

        # Task 4

        # test_pred_dict is a list of dictionaries for the test data which has been classified as positive for heart disease
        test_pred_dict = tree_predict(self.arbre, task2_test_data)

        #############

        ''' I'm going to find a way to make the following code better, i just needed to remove the attributes age and sex to continue with the next part '''

        '''for dictionary in test_pred_dict:
            keys_to_remove = ['age', 'sex']
            for key in keys_to_remove:
                new_pred_dict = dictionary.pop(key)'''

        dict_no_age = []
        for i in test_pred_dict:
            dict_no_age.append({attribute: value for attribute, value in i.items() if attribute != 'age'})

        dict_no_sex = []
        for i in dict_no_age:
            dict_no_sex.append({attribute: value for attribute, value in i.items() if attribute != 'sex'})

        #################

        # converts test_pred_dict into a list of lists
        test_pred_list = dict_to_list(dict_no_sex)

        neg_rules = rules_for_negative_class(self.regles)

        print(neg_rules)
        # print(self.regles)
        print()
        print()
        print(test_pred_list)

        # print(tree_predict(self.arbre, task2_test_data))

        # Task 5

        # Training
        df = pd.read_csv('data/train_continuous.csv')
        task5_train_data = parse_data(df)

        id3_cts = ID3Cts()
        tree_advance = id3_cts.construit_arbre(task5_train_data)

        self.arbre_advance = tree_advance

        # Stats
        print('\n***** Task 5 *****')

        print('max height of the tree:', max_depth(self.arbre_advance))
        print('min height of the tree:', min_depth(self.arbre_advance))
        print('number of leaves in the tree:', get_leaf_count(self.arbre_advance))
        print('average height of the tree:', average_height(self.arbre_advance))

        # Testing
        df = pd.read_csv('data/test_public_continuous.csv')
        task5_test_data = parse_data(df)
        task5_accuracy = test_stats(self.arbre_advance, task5_test_data)
        print('accuracy is :', format(task5_accuracy, '.2%'))

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

    ah = sum(lengths) / len(paths)

    return ah


def test_stats(tree, data):
    """ Test the tree on a test dataset.

        :param NoeudDeDecision tree: the tree to test
        :param list data: test data
        :return: accuracy of the classifications
    """
    success = 0
    for target, inp in data:
        if tree.classifie(inp).split()[-1] == target:
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


def tree_predict(t, data):
    """ get prediction of a dataset using a decision tree, t

        :param NoeudDeDecision t: the tree used to make prediction
        :param list data: data in which to find the prediction
        :return: returns the datapoints which were classified as '1'
    """
    pred_heart_disease = []
    for target, inp in data:
        if t.classifie(inp)[-1] == '1':
            pred_heart_disease.append(inp)

    return pred_heart_disease


def dict_to_list(dictionaries):
    """ transforms a list of dictionaries to a list of lists

        :param dictionaries: a list of dictionaries
        :return: a list of lists
    """
    new_list = []
    for d in dictionaries:
        new_list.append([[attribute, value] for attribute, value in d.items()])

    return new_list


def rules_for_negative_class(rules):
    """ reduces original list of rules of a decision tree to a list containing 
        only the rules which lead to a negative classification

        :param list rules: a list of rules for a decision tree
        :return: a reduced list of rules for a negative classification
    """

    original_rules = rules
    neg_rules = []

    for rule in original_rules:
        if rule[-1] == '0':
            rule.pop()
            neg_rules.append(rule)

    return neg_rules
