"""
Apply Best-Fit Extension to the coarsely selected genes.
"""
import numpy as np
import pandas as pd
from sim import *
from network_inference import best_fit

# each column is the importance of possible regulators
importance_data = pd.read_csv('importance_data.csv', index_col=0)

# all the genes
genes = list(Genes.__members__.keys())

# the training set
initial_state = np.array([1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0])
training_set_list = generate_training_sets_asynchronous(initial_state)

# choose the first 6 genes by coarse selection
L = 6
for g, gene in enumerate(genes):
    imp = importance_data[gene].sort_values(ascending=False)
    potential_regulators = []
    for i in range(L):
        potential_regulators.append(Genes[imp.index[i]])
    print('Target: ', gene)
    print('Regulators by rfBFE:')
    X, y = training_set_list[g]
    print(best_fit(X, y, potential_regulators))
    print('------------------------')
