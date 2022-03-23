"""
Three inference algorithms: reveal, Best-fit and decision tree.
Reference:
[1] Christoph Müssel, Martin Hopfensitz, Hans A. Kestler (2010). BoolNet -- an R package for generation, reconstruction
    and analysis of Boolean networks. Bioinformatics 26(10):1378-1380.
"""

import itertools
from sklearn import tree
import numpy as np
from gene_network import Genes
from sim import state_to_index


def _entropy(X: np.ndarray):
    """
    Compute the entropy of an array X, where each row is a sample.
    :param X: an 1d or 2d array
    :return: 
    """
    if X.ndim == 1:
        data = X
    else:   # 2d
        data = np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize * X.shape[1])))
    u, counts = np.unique(data, return_counts=True)
    p = counts / counts.sum()   # probability of each occurrence
    return -(p * np.log(p)).sum()


def _removeInconsistency(X, y):
    """
    Choose an identical label for each input pattern such that the training set can have an extension
    :param X: input of the training set
    :param y: output of the training set
    :return:  X and y with conflicting labels modified to be consistent
    """
    index_table = [0] * (2 ** len(Genes))
    for i in range(len(y)):
        index = state_to_index(X[i, :])
        if y[i] == 1:
            index_table[index] += 1
        else:
            index_table[index] -= 1
    for i in range(len(y)):
        index = state_to_index(X[i, :])
        if index_table[index] >= 0:
            y[i] = 1
        else:
            y[i] = 0


def reveal(X, y):
    """
    REVEAL algorithm, infer the regulators for a gene given the training set
    :param X: input in 2d array, where each row represents a network state
    :param y: output in a vector, where each element means the state of a gene
    :return: a list containing the regulators, or None if no matching regulators found
    """
    y = np.copy(y)
    _removeInconsistency(X, y)
    for k in range(1, len(Genes) + 1):
        for c in itertools.combinations(Genes, k):  # for each combination
            iX = X[:, c]
            hx = _entropy(iX)
            hxy = _entropy(np.hstack((iX, y.reshape((y.size, 1)))))
            if hx == hxy:
                return set(c)
    return None


def best_fit(X, y, candidate_genes=Genes):
    """
    Best-fit extension algorithm, infer the regulators for a gene given the training set
    :param X: input in 2d array, where each row represents a network state
    :param y: output in a vector, where each element means the state of a gene
    :param candidate_genes: the candidates for regulator test
    :return: a list containing the regulators
    """
    min_error = len(y) + 1
    min_c = None    # the regulator list corresponding to the minimum classification error
    for k in range(1, len(candidate_genes) + 1):
        for c in itertools.combinations(candidate_genes, k):
            count_0 = [0] * (2 ** k)
            count_1 = [0] * (2 ** k)
            data = X[:, c]  # enum can be used as numbers
            for i in range(data.shape[0]):
                index = state_to_index(data[i, :])
                if y[i] == 0:
                    count_0[index] += 1     # default weight is 1
                else:
                    count_1[index] += 1
            # now for this regulator combination c, we choose its "true" label for each input to minimize
            # misclassification errors
            error = 0
            for i in range(len(count_0)):
                if count_0[i] > count_1[i]:     # true label is 0, count_1 is misclassified by our function
                    error += count_1[i]
                else:
                    error += count_0[i]
            if error < min_error:
                min_error = error
                min_c = c
            if min_error == 0:  # choose the one with minimum k, once min error is 0, terminate enumeration
                return set(min_c)
    return set(min_c)


def decision_tree_infer(X, y, importance_threshold=0):
    """
    Decision tree for Boolean network inference (DTBNI), infer the regulators for a gene given the training set 
    :param X: input in 2d array, where each row represents a network state
    :param y: output in a vector, where each element means the state of a gene
    :param importance_threshold: critetiorn for the regulator selection. 0: choose the ones with non-zero importance.
    :return: a list containing the regulators
    """
    y = np.copy(y)
    _removeInconsistency(X, y)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    feature_importances = clf.feature_importances_
    index_array = np.argsort(feature_importances)[::-1]     # sort in descending order
    c = []
    for index in index_array:
        if feature_importances[index] > 0:
            c.append(Genes(index))
        else:
            break
    return set(c)




