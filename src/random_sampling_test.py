from sim import *
from network_inference import *


def random_sampling(training_set_list, q):
    """
    Generate a random sampling of size q of the training set
    :param training_set_list: a list of n training sets for the n genes, each set in form (X, y)
    :return: the sampled training set list
    """
    sampled_training_set_list = []
    for ts in training_set_list:
        X = ts[0]
        y = ts[1]
        choice = np.random.choice(len(y), q, replace=False)
        sampled_training_set_list.append((X[choice, :], y[choice] ))
    return sampled_training_set_list


def random_test_without_noise():
    q_list = [5, 10, 20, 40, 80, 160, 320]
    num_repetitions = 100
    counts = np.empty((len(q_list), num_repetitions, 3), dtype=int)  # count the number of genes that has been inferred
    methods = [reveal, best_fit, decision_tree_infer]
    complete_training_set_list = generate_complete_training_set()
    for j, q in enumerate(q_list):
        print("Now processing q = ", q)
        for r in range(num_repetitions):
            scores = [0] * 3
            sampled_training_set_list = random_sampling(complete_training_set_list, q)
            for g in Genes:  # infer regulators for each gene with three methods
                for i, method in enumerate(methods):
                    c = method(*sampled_training_set_list[g])
                    if c == Regulators[g]:
                        scores[i] += 1  # once a method gives the true regulators for a gene, wins one point
            counts[j, r, :] = scores

    np.save("counts_without_noise", counts)


def add_noise(training_set_list, probability=0.1):
    """
    Add noise to the output y in each training set (flip 0 and 1) according to the given probability
    :param training_set_list: a list containing training set for each gene in form (X, y)
    :param probability: flip the state of each sample' output y with this probability
    :return: void
    """
    for training_set in training_set_list:
        _, y = training_set
        random_filter = np.random.random((len(y), )) < probability
        y[random_filter] = 1 - y[random_filter]


def random_test_with_noise(probability=0.1):
    """
    Random sampling of the complete training set (the whole state space) and add noise, then infer the regulators.
    :param probability: the probability for flipping the output to mimic noise effect
    :return: void
    """
    q_list = [5, 10, 20, 50, 100, 300, 500]
    num_repetitions = 100
    counts = np.empty((len(q_list), num_repetitions, 3), dtype=int)  # count the number of genes that has been inferred
    methods = [reveal, best_fit, decision_tree_infer]
    complete_training_set_list = generate_complete_training_set()
    for j, q in enumerate(q_list):
        print("Now processing q = ", q)
        for r in range(num_repetitions):
            scores = [0] * 3
            sampled_training_set_list = random_sampling(complete_training_set_list, q)
            add_noise(sampled_training_set_list, probability)
            for g in Genes:  # infer regulators for each gene with three methods
                for i, method in enumerate(methods):
                    c = method(*sampled_training_set_list[g])
                    if c == Regulators[g]:
                        scores[i] += 1  # once a method gives the true regulators for a gene, wins one point
            counts[j, r, :] = scores

    np.save("counts_with_noise", counts)


"""
Perform regulator inference with REVEAL, Best-fit and DT algorithms on a subset of the state space.
- no noise --> result stored in counts_without_noise.npy (case 2 in the ICCA paper)
- with noise to simulate measurement errors --> result stored in counts_with_noise.npy
The performance result is a 3d array (dataset size, repetition, methods). Each element is the number of genes whose 
regulators have been accurately identified.
"""
if __name__ == "__main__":
    random_test_without_noise()

# plot the results

