from math import log
from statistics import mean
from .noeud_de_decision_cts import NoeudDeDecisionCts


class ID3Cts:
    """ Algorithme ID3. 

        This is an updated version from the one in the book (Intelligence Artificielle par la pratique).
        Specifically, in construit_arbre_recur(), if donnees == [] (line 70), it returns a terminal node with the predominant class of the dataset -- as computed in construit_arbre() -- instead of returning None.
        Moreover, the predominant class is also passed as a parameter to NoeudDeDecision().
    """

    def construit_arbre(self, donnees):
        """ Construit un arbre de décision à partir des données d'apprentissage.

            :param list donnees: les données d'apprentissage\
            ``[classe, {attribut -> valeur}, ...]``.
            :return: une instance de NoeudDeDecision correspondant à la racine de\
            l'arbre de décision.
        """

        # Nous devons extraire les domaines de valeur des 
        # attributs, puisqu'ils sont nécessaires pour 
        # construire l'arbre.
        attributs = self.get_attributes(donnees)

        # Find the predominant class
        classes = set([row[0] for row in donnees])
        # print(classes)
        predominant_class_counter = -1
        for c in classes:
            # print([row[0] for row in donnees].count(c))
            if [row[0] for row in donnees].count(c) >= predominant_class_counter:
                predominant_class_counter = [row[0] for row in donnees].count(c)
                predominant_class = c
        # print(predominant_class)

        arbre = self.construit_arbre_recur(donnees, attributs, predominant_class)

        return arbre

    def get_attributes(self, donnees):
        """ Extracts the attributes and their possible values.

            :param list donnees: data from which attributes are extracted.
            :return: dictionary mapping each attribute to a list of its possible values.
        """
        attributs = {}
        for donnee in donnees:
            for attribut, valeur in donnee[1].items():
                valeurs = attributs.get(attribut)
                if valeurs is None:
                    valeurs = set()
                    attributs[attribut] = valeurs
                valeurs.add(float(valeur))
        return attributs

    def construit_arbre_recur(self, donnees, attributs, predominant_class):
        """ Construit rédurcivement un arbre de décision à partir 
            des données d'apprentissage et d'un dictionnaire liant
            les attributs à la liste de leurs valeurs possibles.

            :param list donnees: les données d'apprentissage\
            ``[classe, {attribut -> valeur}, ...]``.
            :param attributs: un dictionnaire qui associe chaque\
            attribut A à son domaine de valeurs a_j.
            :return: une instance de NoeudDeDecision correspondant à la racine de\
            l'arbre de décision.
        """

        def classe_unique(donnees):
            """ Vérifie que toutes les données appartiennent à la même classe. """

            if len(donnees) == 0:
                return True
            premiere_classe = donnees[0][0]
            for donnee in donnees:
                if donnee[0] != premiere_classe:
                    return False
            return True

        if donnees == []:
            return NoeudDeDecisionCts(None, [str(predominant_class), dict()], str(predominant_class))

        # Si toutes les données restantes font partie de la même classe,
        # on peut retourner un noeud terminal.         
        elif classe_unique(donnees):
            return NoeudDeDecisionCts(None, donnees, str(predominant_class))

        else:
            # Sélectionne l'attribut qui réduit au maximum l'entropie.
            entropy_per_attribute = [(self.threshold_smallest_entropy(donnees, attribut, list(attributs[attribut])) +
                                      [attribut]) for attribut in attributs if len(attributs[attribut]) > 1]

            threshold, entropy, attribut = min(entropy_per_attribute, key=lambda h_a: h_a[1])

            # Crée les sous-arbres de manière récursive.
            partitions = self.partitionne(donnees, attribut, threshold)

            enfants = {}
            for valeur, partition in partitions.items():
                attributs_restants = self.get_attributes(partition)
                enfants[valeur] = self.construit_arbre_recur(partition,
                                                             attributs_restants,
                                                             predominant_class)

            return NoeudDeDecisionCts(attribut, donnees, str(predominant_class), enfants)

    def partitionne(self, donnees, attribut, threshold):
        """ Partitionne les données sur les valeurs a_j de l'attribut A.

            :param list donnees: les données à partitioner.
            :param attribut: l'attribut A de partitionnement.
            :param threshold: the threshold to split the data.
            :return: un dictionnaire qui représente 2 subtrees ayant des valeurs de A\
            plus grandes ou plus petites que threshold.
        """

        left, right = self.split_according_to_threshold(donnees, attribut, threshold)

        partitions = {"<= " + str(threshold): left, "> " + str(threshold): right}

        return partitions

    def threshold_smallest_entropy(self, donnees, attribut, attr_values):
        """ Finds a threshold that minimises the entropy and the resulting entopy.

            :param list donnees: data to partition
            :param attribut: attribute to consider
            :param list attr_values: possible values for the attribute
            :return: the treshold to use and the resulting entropy as a list of two elements
        """
        sorted_vals = sorted(set(attr_values))
        possible_thresholds = [mean(x) for x in zip(sorted_vals[1:], sorted_vals[:-1])]
        possible_entropies = [self.entropy_A(donnees, attribut, threshold) for threshold in possible_thresholds]

        return list(min(zip(possible_thresholds, possible_entropies), key=lambda pair: pair[1]))

    def entropy_A(self, donnees, attribute, threshold):
        """ The entropy resulting from splitting the data according to attribute at the specified threshold.

            :param list donnees: les données d'apprentissage.
            :param attribute: the attribute A.
            :param threshold: the threshold used to split the data.
            :return: entropy
        """
        # Nombre de données.
        nombre_donnees = len(donnees)

        # Permet d'éviter les divisions par 0.
        if nombre_donnees == 0:
            return 0.0

        # Split data into two lists acording to threshold and attribute
        left, right = self.split_according_to_threshold(donnees, attribute, threshold)

        entropy = (len(left) * self.entropy(left) + len(right) * self.entropy(right)) / nombre_donnees
        return entropy

    def split_according_to_threshold(self, data, attribute, threshold):
        """ Splits data into two lists according to an attribute threshold.

            :param list data: list of data points.
            :param attribute: the attribute used in the split.
            :param int threshold: the threshold to split.
            :return: two lists for smaller/larger than the threshold.
        """
        left = list()
        right = list()
        for donnee in data:
            if float(donnee[1][attribute]) <= threshold:
                left.append(donnee)
            else:
                right.append(donnee)

        return left, right

    def entropy(self, data):
        """ Computes the Shannon entropy of the data according to the classes.

            :param list data: list of data. First element of each datum is the class.
            :return: entropy
        """
        class_0 = [x for x in data if float(x[0]) == 0]

        p0 = len(class_0) / len(data)
        p1 = 1 - p0

        if p0 * p1 == 0:
            return 0

        return -(p0 * log(p0) + p1 * log(p1))
