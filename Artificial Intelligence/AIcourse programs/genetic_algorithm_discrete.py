"""
Simple Genetic Algorithm From Scratch in Python
by Jason Brownlee
https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/

Genetic Algorithm for OneMax

Next we will apply the genetic algorithm to a binary string-based
optimization problem.
The problem is called OneMax and evaluates a binary string based
on the number of 1s in the string. For example, a bitstring with
a length of 20 bits will have a score of 20 for a string of all 1s.
Given we have implemented the genetic algorithm to minimize the
objective function, we can add a negative sign to this evaluation
so that large positive values become large negative values.
"""

# genetic algorithm search of the one max optimization problem
from numpy.random import randint
from numpy.random import rand


"""
objective function
The genetic algorithm try to minimize the objective function, 
thus we add a negative sign to this evaluation so that 
large positive values become large negative values.
"""
def onemax(x):
    """
    generic objective function to get a fitness score,
    which we will minimize.
    :param x: bitstring of integer values
    :return:  fitness score
    """
    return -sum(x)


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
def genetic_algorithm(objective, n_bits, n_iter, n_pop,
                      r_cross, r_mut):
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
    """
    The first step is to create a population of 
    random bitstrings. We will use integer values.
    We generate an array of integer values in a  
    range using the randint() function.
    We represent a candidate solution as a list
    """
    pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
    # keep track of best solution
    best, best_eval = 0, objective(pop[0])
    # enumerate generations
    for gen in range(n_iter):
        # evaluate all candidates in the population
        scores = [objective(c) for c in pop]
        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
        # select parents for each position in the population to create a list of parents.
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


if __name__ == '__main__':
    # define the total number of iterations
    n_iter = 100
    # number of bits to be used
    n_bits = 20
    # define the population size
    n_pop = 100
    # crossover rate (hyperparameter)
    r_cross = 0.9
    # mutation rate (hyperparameter)
    r_mut = 1.0 / float(n_bits)
    # perform the genetic algorithm search
    best, score = genetic_algorithm(onemax, n_bits, n_iter, n_pop, r_cross, r_mut)
    print('Done!')
    print('f(%s) = %f' % (best, score))

