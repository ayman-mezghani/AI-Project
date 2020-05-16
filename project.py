from moteur_id3.id3 import ID3
import pandas as pd


class ResultValues():

    def __init__(self):
        # Do computations here
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
                return max(children_depth)+1

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

                # Task 1
        df = pd.read_csv('data/train_bin.csv')
        # df.info()
        # print(list(df)[:len(list(df)) - 1])
        data = parse_data(df)

        id3 = ID3()
        tree = id3.construit_arbre(data)

        print('max height of the tree:', max_depth(tree))
        print('min height of the tree:', min_depth(tree))
        print('number of leaves in the tree:', get_leaf_count(tree))

        self.arbre = tree

        # Task 3
        self.faits_initiaux = None
        self.regles = None
        # Task 5
        self.arbre_advance = None

    def get_results(self):
        return [self.arbre, self.faits_initiaux, self.regles, self.arbre_advance]
