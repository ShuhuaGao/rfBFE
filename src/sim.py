"""
Simulate the network under various circumstances to generate data.
"""

from gene_network import *


def state_to_index(s):
    """
    For convenience, in many cases we use an index to represent a state by interpreting the state as a binary representation
    of an integer (MSB first). For example, state (0, 1, 1, 0, 1, 0) is assigned index 26.
    :param s: network state
    :return: index representation of the state
    """
    index = 0
    for i in s:
        index <<= 1
        index |= i
    return index


def index_to_state(index):
    """
    Given an index integer, explain its binary representation (MSB first) as a network state
    :param index: an integer in range [1, 2^n - 1] (both inclusive)
    :return: the state as a vector
    """
    n = len(Genes)
    return np.array([int(x) for x in format(index, "0{0}b".format(n))])


def generate_all_network_states():
    """
    Generate all the 2^n possible states of the network. 
    The row index in the returned matrix is equal to the binary presentation given by the state.
    :return: a matrix with each row as a state, shape: (2^n, n)
    """
    n = len(Genes)  # number of genes
    xi = tuple([[0, 1]]) * n
    grid = np.meshgrid(*xi)
    all_states = np.empty((2**n, n), dtype=int)
    for i in range(len(grid)):
        all_states[:, i] = grid[i].flatten()
    return all_states


def generate_all_asynchronous_state_transitions():
    """
    Compute all the possible state transitions in asynchronous strategy.
    :return: a list of length 2^n. Each index represents the current state and each element is n-by-n matrix representing
    the n possibly successive states with one row as one possibility. Here, n is the number of genes in the network.
    """
    n = len(Genes)
    transitions = []
    all_states = generate_all_network_states()
    for i in range(2 ** n):
        s = all_states[i]
        new_states = np.empty((n, n), dtype=int)
        for g in Genes:
            new_states[g, :] = update(s, g)    # new state resulted from updating g
        transitions.append(new_states)
    return transitions


def generate_all_synchronous_state_transitions():
    """
    Compute all the 2^n state transitions in synchronous updating scheme, where n is the number of genes.
    :return: a matrix of 2^n-by-n, whose row index represents each current state and whose row represents the 
    successor state after synchronous update. 
    """
    n = len(Genes)
    transitions = np.empty((2**n, n), dtype=int)
    all_states = generate_all_network_states()
    for i in range(2 ** n):
        s = all_states[i]
        transitions[i] = update(s, None)     # synchronous, a deterministic follow-up state
    return transitions


def generate_all_IO_pairs(g):
    """
    For all the 2^n input network states, generate the output state of gene g.
    :param g: a gene
    :return: a vector, whose index is the input network state and the element is the output state (0/1) of gene g
    """
    n = len(Genes)
    outputs = np.empty((n, 1), dtype=int)
    all_states = generate_all_network_states()
    for i in range(2 ** n):
        s = all_states[i]
        outputs[i] = update(s, g)
    return outputs


def generate_complete_training_set():
    """
    For each of all the 2^n states, we can acquire an training sample for each gene. Therefore, a complete 
    training set for a gene will include 2^n distinct samples. Here, n is the number of genes.
    :return: a list containing the complete training set for each gene. Each training set is (X, y), where X is a 
    matrix of size 2^n-by-n and y is a vector including the 2^n output states for a certain gene.
    """
    all_states = generate_all_network_states()
    n_training_set = []
    for g in Genes:
        X = []
        y = []
        for s in all_states:
            new_s = update(s, g)
            X.append(s)
            y.append(new_s[g])
        n_training_set.append((np.array(X), np.array(y)))
    return n_training_set



def generate_synchronous_trajectory(initial_state):
    """
    Simulate the network starting from a given initial state in the synchronous strategy
    :param initial_state: initial state of the network
    :return: a trajectory in matrix from, where each row denotes a state
    """
    trajectory = [initial_state]
    state_index_set = {state_to_index(initial_state)}    # if a state reoccurs, an attractor or fixed point is
    # reached, stop.
    s = initial_state
    while True:
        new_s = update(s)   # synchronous
        new_s_index = state_to_index(new_s)
        if new_s_index in state_index_set:
            break
        trajectory.append(new_s)
        state_index_set.add(new_s_index)
        s = new_s
    return np.array(trajectory)


def generate_all_distinct_states_asynchronous(initial_state):
    """
    Simulate the network asynchronously from the given initial state and get all the possible distinct states in this 
    process.
    :param initial_state: the initial network state for simulation
    :return: all the distinct network states in the simulation, stored in a 2d array with each row as a state
    """
    state_index_set = {state_to_index(initial_state)}
    distinct_states = [initial_state]
    buffer = [initial_state]
    while buffer:
        s = buffer.pop()
        for g in Genes:
            new_s = update(s, g)
            index = state_to_index(new_s)
            if index not in state_index_set:    # a new distinct state is found
                distinct_states.append(new_s)
                state_index_set.add(index)
                buffer.append(new_s)
    return np.array(distinct_states)


def is_asynchronous_fixed_point(s):
    """
    Check whether a state is a fixed point in asynchronous manner.
    :param s: a state
    :return: true if it is a fixed point, i.e., stay the same no matter which gene is updated
    """
    for g in Genes:
        new_s = update(s, g)
        if not all(new_s == s):
            return False
    return True


def generate_training_sets_asynchronous(initial_state):
    """
    First simulate the network asynchronously starting from the initial state and get the distinct states.
    For each distinct state, we form an input-output pair for gene g by updating input state by gene g as long as 
    gene g is changed, i.e., the network state is changed through asynchronous updating using gene g.
    :param initial_state: the initial network state
    :return: a list of training sets for each gene. Each training set is (X, y), where X is a 2d array input and y is 
    an output vector. 
    """
    states = generate_all_distinct_states_asynchronous(initial_state)
    training_sets = []
    for g in Genes:
        X = []
        y = []
        for s in states:
            new_s = update(s, g)

            X.append(s)
            y.append(new_s[g])
        training_sets.append((np.array(X), np.array(y)))
    return training_sets


if __name__ == '__main__':
    # generate the data states from a given initial state for the myeloid network
    initial_state = np.array([1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0])
    states = generate_all_distinct_states_asynchronous(initial_state)
    print(states.shape)
    np.savez('myeloid_data', initial_state=initial_state, states=states)