"""
Case 3 for ICCA 2018: rfBFE method
"""
from network_inference import *
from sim import *
import numpy as np
from feature_selection import *
import pandas as pd


if __name__ == "__main__":
    initial_state = np.array([1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0])
    rfBFE_score = 0
    training_set_list = generate_training_sets_asynchronous(initial_state)
    genes = list(Genes.__members__.keys())
    df = pd.DataFrame()
    for g, ts in enumerate(training_set_list):
        print('Target: ', Genes(g))
        X, y = ts
        rf = random_forest_classification(X, y)
        print('Feature importance:')
        imp = obtain_feature_importances(rf, genes)
        print(imp)
        print('-----------------------------')
        df[Genes(g).name] = imp
    df.to_csv('importance_data.csv')
