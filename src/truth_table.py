"""
Get the truth table for the myeloid differentiation based on the regulators we have identified
"""

from gene_network import Genes
from sim import generate_training_sets_asynchronous
import numpy as np

regulators_BDT = {Genes.GATA2: (Genes.FOG1, Genes.PU1),
                  Genes.GATA1: Genes.PU1,
                  Genes.FOG1: Genes.GATA1,
                  Genes.EKLF: (Genes.GATA1, Genes.Fli1),
                  Genes.Fli1: (Genes.GATA1, Genes.EKLF),
                  Genes.SCL: Genes.GATA1,
                  Genes.CEBPa: (Genes.FOG1, Genes.SCL),
                  Genes.PU1: (Genes.GATA2, Genes.PU1),
                  Genes.cJun: (Genes.Gfi1, Genes.PU1),
                  Genes.EgrNab: (Genes.PU1, Genes.cJun, Genes.Gfi1),
                  Genes.Gfi1: (Genes.CEBPa, Genes.EgrNab)}


initial_state = np.array([1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0])
training_set_list = generate_training_sets_asynchronous(initial_state)

X, y = training_set_list[0]
X = X[:, [Genes.FOG1, Genes.PU1]]


def get_truth_table(X, y, regulators):
    X = X[:, regulators]
    entries = set()
    truth_table = []
    for i, entry in enumerate(X):
        if not np.isscalar(entry):
            entry = tuple(entry)
        if entry not in entries:
            truth_table.append([entry, y[i]])
            entries.add(entry)
    return truth_table


truth_table_dict = {}
for g in Genes:
    X, y = training_set_list[g]
    truth_table_dict[g] = get_truth_table(X, y, regulators_BDT[g])


for g in truth_table_dict:
    print(regulators_BDT[g], ' -> ', g)
    for entry in truth_table_dict[g]:
        print(entry)
    print('---------------------------------\n')