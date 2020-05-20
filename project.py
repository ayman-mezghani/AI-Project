from moteur_id3.id3 import ID3
from csv import DictReader
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

       #defining the initial facts 
        with open('data/train_bin.csv', 'r') as read_obj:
            # pass the file object to DictReader() to get the DictReader object
            csv_dict_reader = DictReader(read_obj)
            # iterate over each line as a ordered dictionary
            for row in csv_dict_reader:
                # row variable is a dictionary that represents a row in csv

                # defining the sex of the person
                if row['sex'] == 1:
                    row['sex'] = 'male'
                else:
                    row['sex'] = 'female'
            
                # define the chest pain type 
                if row['cp'] == 0:
                    row['cp'] = 'typical angina'
                elif row['cp'] == 1:
                    row['cp'] = 'atypical angina'
                elif row['cp'] == 2:
                    row['cp'] = 'non-anginal pain'
                else:
                    row['cp'] = 'asymptomatic'
            
                #defining whether fasting blood sugar is greater than 120 mg/dl
                if row['fbs'] == 1:
                    row['fbs'] = 'true'
                else:
                    row['fbs'] = 'false'
            
                #defining resting electrocardiographic results
                if row['restecg'] == 0:
                    row['restecg'] = 'normal'
                elif row['restecg'] == 1:
                    row['restecg'] = 'has ST-T wave abnormality'
                else:
                    row['restecg'] = "shows probable or definite left ventricular hypertrophy by Estes' criteria"
            
                #defining whether a person has exercise induced angina
                if row['exang'] == 1:
                    row['exang'] = 'true'
                else:
                    row['exang'] = 'false'
            
                #defining slope of the peak exercise ST segment
                if row['slope'] == 0:
                    row['slope'] = 'unsloping'
                elif row['slope'] == 1:
                    row['slope'] = 'flat'
                else:
                    row['slope'] = 'downsloping'
            
                #defining thal
                if row['thal'] == 0:
                    row['thal'] = 'missing'
                elif row['thal'] == 1:
                    row['thal'] = 'normal'
                elif row['thal'] == 2:
                    row['thal'] = 'fixed defect'
                else:
                    row['thal'] = 'reversable defect'
            
                #defining whether the person has heart disease
                if row['target'] == 0:
                    row['target'] = 'no heart disease'
                else:
                    row['target'] = 'has heart disease'


                self.faits_initiaux = row 


        self.regles = None


        # Task 5
        self.arbre_advance = None

    def get_results(self):
        return [self.arbre, self.faits_initiaux, self.regles, self.arbre_advance]
