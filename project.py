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

        print('\n***** Task 4 *****')

        positive_count = 0
        saved_count = 0
        for case in task2_test_data:
            # if the case tests positive

            if tree.classifie(case[1]).split()[-1] == '1':
                positive_count += 1
                explanation = explain_prediction(self.regles, [[x, y] for x, y in case[1].items()])
                # print(explanation)
                if explanation.split()[-1] != '[]':
                    saved_count += 1

        print('saved', saved_count, 'out of', positive_count, 'positive cases.')
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
        print('average height of the tree:', "{:.2f}".format(average_height(self.arbre_advance)))

        # Testing
        df = pd.read_csv('data/test_public_continuous.csv')
        task5_test_data = parse_data(df)
        task5_accuracy = test_stats(self.arbre_advance, task5_test_data)
        print('accuracy is:', format(task5_accuracy, '.2%'))
        # print(self.arbre_advance.classifie(task5_train_data[0][1]))

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


def explain_prediction(rules, datapoint, c=2):
    """ get prediction explanation for the datapoind based on the rules.

        :param list rules: the list of rules
        :param list datapoint: the datapoint attributes
        :param int c: max cost of treatmen to suggest
        :param
        :return: the explanation of the prediction
    """
    for rule in rules:
        if all(i in datapoint for i in rule[:-1]):
            res = str(rule[:-1]) + " => " + rule[-1]
            if rule[-1] == '1':
                neg_rules = rules_for_negative_class(rules)
                treatment = suggest_treatement(neg_rules, datapoint, 2)
                if len(treatment) > 0:
                    res += '\nWe sugenst these changes: ' + str(treatment)
                else:
                    res += "\nWe don't have any treatments with cost less than " + str(c) + " to suggest."
            return res
    return "No explanation found. Ask a real doctor!"


def rules_for_negative_class(rules):
    """ reduces original list of rules of a decision tree to a list containing 
        only the rules which lead to a negative classification

        :param list rules: a list of rules for a decision tree
        :return: a reduced list of rules for a negative classification
    """
    negative_rules = [x for x in rules if x[-1] == '0']

    return negative_rules


def suggest_treatement(rules, datapoint, c):
    """ Suggests the treatment with a cost smaller or equal than c

        :param list rules: the list of negative rules
        :param list datapoint: the datapoint attributes
        :param int c: maximum cost of the treatment
        :return: a treatment costing less than c if there exists one
    """
    treatments = [[e for e in rule[:-1] if e not in datapoint] for rule in rules]

    valid_treatments = [t for t in treatments if 'sex' not in str(t) and 'age' not in str(t)]

    candidate = min(valid_treatments, key=len)
    if len(candidate) <= c:
        return candidate
    else:
        return []
