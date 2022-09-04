"""
Simple Genetic Algorithm From Scratch in Python
by Jason Brownlee
https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/

genetic algorithm search for continuous function optimization
"""

from numpy.random import randint
from numpy.random import rand
import random as rnd
import numpy as np
from matplotlib import pyplot
import graph_module as gm
import graph_examples as ge

seed = 2022
rnd.seed(seed)
np.random.seed(seed)


# objective function
def objective(x):
    """
    define the minimization function as
    f(x) = x[0]**4 - 4 * x[0]**3 - 2*x[0]**2 + 5*x[0] + 9
    :param x: input variables
    :return:  fitness score
    """
    return x[0]**4 - 4 * x[0]**3 - 2*x[0]**2 + 5*x[0] + 9


# decode bitstring to numbers
def decode(bounds, n_bits, bitstring):
    """
    to decode the bit strings to numbers prior to
    evaluating each with the objective function.
    First decode each substring to an integer,
    then scale the integer to the desired range.
    This will give a vector of values in the range that can
    then be provided to the objective function for evaluation.
    :param bounds: bounds of the function
    :param n_bits: number of bits per variable
    :param bitstring: bitstring value
    :return: list of decoded real values
    """
    decoded = list()
    largest = 2**n_bits
    for i in range(len(bounds)):
        # extract the substring
        start, end = i * n_bits, (i * n_bits)+n_bits
        substring = bitstring[start:end]
        # convert bitstring to a string of chars
        chars = ''.join([str(s) for s in substring])
        # convert string to integer
        integer = int(chars, 2)
        # scale integer to desired range
        value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
        # store
        decoded.append(value)
    return decoded


# tournament selection
def selection(pop, scores, k=3):
    """
    The tournament selection procedure takes
    the population and returns one selected parent.
    The k value is fixed at 3 by default.
    :param pop: population
    :param scores: scores
    :param k: number of candidates from the population
    :return: selected parent
    """
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    """
    implements crossover using a draw of a random number
    in the range [0,1] to determine if crossover is performed,
    then selecting a valid split point if crossover is to
    be performed.
    The crossover rate is a hyperparameter that determines
    whether crossover is performed or not, and if not,
    the parents are copied into the next generation.
    It is a probability and typically has a large value
    close to 1.0.
    :param p1: parent1
    :param p2: parent2
    :param r_cross: crossover rate
    :return: two children
    """
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1)-2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]


# mutation operator
def mutation(bitstring, r_mut):
    """
    This procedure simply flips bits with a low probability
    controlled by the “r_mut” hyperparameter.
    :param bitstring:
    :param r_mut: mutation rate
    :return: mutated bitstring
    """
    for i in range(len(bitstring)):
        # check for a mutation
        if rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]


# genetic algorithm
def genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
    """
    This function takes the name of the objective function
    and the hyperparameters of the search, and returns
    the best solution found during the search
    :param objective: the objective function
    :param n_bits: number of bits in a single candidate solution
    :param n_iter: number of iterations
    :param n_pop: population size
    :param r_cross: crossover hyperparameter
    :param r_mut: mutation hyperparameter
    :return: best solution found during the search
    """
    # initial population of random bitstring
    """
    We generate an array of integer values in a  
    range using the randint() function.
    We represent a candidate solution as a list
    We need to ensure that the initial population 
    creates random bitstrings that are large enough.
    """
    pop = [randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]
    # keep track of best solution
    best, best_eval = 0, objective(decode(bounds, n_bits, pop[0]))
    # enumerate generations
    for gen in range(n_iter):
        # decode population
        decoded = [decode(bounds, n_bits, p) for p in pop]
        # then evaluate the decoded version of the population.
        scores = [objective(d) for d in decoded]
        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %f" % (gen,  decoded[i], scores[i]))
                lsolutions.append((decoded[i][0], scores[i]))
        # select parents for each position in the population
        # to create a list of parents.
        selected = [selection(pop, scores) for _ in range(n_pop)]
        """
        Create the next generation:
        loop over the list of parents and 
        create a list of children to be used as 
        the next generation, calling the crossover 
        and mutation functions as needed
        """
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
    return [best, best_eval]


def draw_graph1():
    # sample input range uniformly at 0.1 increments
    [r_min, r_max] = bounds[0]
    inputs = np.arange(r_min, r_max, 0.1)
    # compute targets
    results = [objective([x]) for x in inputs]
    # create a line plot of input vs result
    pyplot.plot(inputs, results)
    # define optimal input value
    x_optima = 3.20  # 0.0
    # print('x_optima:', x_optima)
    # draw a vertical line at the optimal input
    pyplot.axvline(x=x_optima, ls='--', color='red')
    # show the plot
    pyplot.title('Objective function')
    gm.save_figure(example['graph_file'])
    pyplot.show()


def draw_graph2():
    # sample input range uniformly at 0.1 increments
    [r_min, r_max] = bounds[0]
    inputs = np.arange(r_min, r_max, 0.1)
    # compute targets
    results = [objective([x]) for x in inputs]
    # create a line plot of input vs result
    pyplot.plot(inputs, results)
    # define optimal input value
    x_optima = 3.20  # 0.0
    # print('x_optima:', x_optima)
    # draw a vertical line at the optimal input
    pyplot.axvline(x=x_optima, ls='--', color='red')

    # plot the sample as black circles
    pyplot.plot(lsolutions, [objective(x) for x in lsolutions], 'o', color='black')
    # show the plot
    pyplot.title('Tested solutions')
    gm.save_figure(example['path_file'])
    pyplot.show()


if __name__ == '__main__':
    # f(x) = x[0]**4 - 4 * x[0]**3 - 2*x[0]**2 + 5*x[0] + 9
    example = ge.example(10)  # genetic_algorithm
    lsolutions = []
    # define the bounds of each input variable.
    bounds = example['bounds']  # [[-5.0, 5.0]]
    draw_graph1()
    # define the total iterations
    n_iter = example['n_iter']  # 100
    # number of bits per input variable to the objective function
    # (hyperparameter)
    # The actual bit string will be (16 * 1) = 16 bits,
    n_bits = example['n_bits']  # 16
    # define the population size
    n_pop = example['n_pop']  # 100
    # crossover rate
    r_cross = example['r_cross']  # 0.9
    # mutation rate
    r_mut = example['r_mut']  # 1.0 / (float(n_bits) * len(bounds))
    # perform the genetic algorithm search
    best, score = genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut)
    print('Done!')
    decoded = decode(bounds, n_bits, best)
    print('f(%s) = %f' % (decoded, score))
    # print('\nlist of solutions:', lsolutions)
    draw_graph2()
