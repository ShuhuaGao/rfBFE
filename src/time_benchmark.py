""""
Measure the performance of these three algorithms in different cases
"""
from sim import *
from network_inference import *
from timeit import default_timer as timer


# the whole state space with no errors
complete_training_set = generate_complete_training_set()


def test_reveal(training_set):
    for ts in training_set:
        c = reveal(*ts)


def test_best_fit(training_set):
    for ts in training_set:
        c = best_fit(*ts)


def test_decision_tree(training_set):
    for ts in training_set:
        c = decision_tree_infer(*ts, importance_threshold=0)


if __name__ == "__main__":
    methods = {"reveal": test_reveal, "BFE": test_best_fit, "rfBFE": test_decision_tree}
    for m in methods.keys():
        print(f"- Running {m}")
        ts = timer()
        methods[m](complete_training_set)
        print(f"  Took time {timer() - ts} seconds.")
