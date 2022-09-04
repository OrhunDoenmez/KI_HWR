"""
Simulated Annealing From Scratch in Python by Jason Brownlee
https://machinelearningmastery.com/simulated-annealing-from-scratch-in-python/#:~:text=Simulated%20Annealing%20is%20a%20stochastic,algorithms%20do%20not%20operate%20well.

"""

# simulated annealing search of a one-dimensional objective function
from numpy import asarray
from numpy import exp
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed
import numpy as np
from matplotlib import pyplot
from time import time
from typing import List
import graph_module as gm
import graph_examples as ge
from pprint import pprint

lsolutions = list()


# objective function
def objective(x):
	# return x[0]**4 - 4 * x[0]**3 - 2*x[0]**2 + 5*x[0] + 9 # function 2
	return x[0]**2.0


# simulated annealing algorithm
def simulated_annealing(objective, bounds, n_iterations, step_size, temp):
	# generate an initial point
	best = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	print('initial point:', best[0])
	# evaluate the initial point
	best_eval = objective(best)
	print('initial point evaluation:', best_eval)
	# current working solution
	curr, curr_eval = best, best_eval
	# lsolutions = list()
	lsolutions.append(curr)
	print('Solution tested:', curr[0])
	scores = list()
	# run the algorithm
	for i in range(n_iterations):
		print('iteration=', i)
		# take a step
		candidate = curr + randn(len(bounds)) * step_size
		# evaluate candidate point
		candidate_eval = objective(candidate)
		# check for new best solution
		if candidate_eval < best_eval:
			# store new best point
			best, best_eval = candidate, candidate_eval
			# keep track of scores
			scores.append(best_eval)
			# report progress
			print('>%d f(%s) = %.5f' % (i, best, best_eval))
		# difference between candidate and current point evaluation
		diff = candidate_eval - curr_eval
		# calculate temperature for current epoch
		t = temp / float(i + 1)
		# calculate metropolis acceptance criterion
		# At each step in the search, a new state is proposed and
		# either accepted or rejected according to a dynamically
		# calculated probability, called the acceptance criteria
		metropolis = exp(-diff / t)
		# check if we should keep the new point
		if diff < 0 or rand() < metropolis:
			# store the new current point
			curr, curr_eval = candidate, candidate_eval
			lsolutions.append(curr)
			print('Solution tested:', curr[0])
	return [best, best_eval, scores]


if __name__ == '__main__':
	example = ge.example(7)  # ex.7 func.1 # ex.8 func.2
	pprint(example)
	begin = time()
	# define range for input
	r_min, r_max = -5.0, 5.0  # function 1
	# sample input range uniformly at 0.1 increments
	inputs = np.arange(r_min, r_max, 0.1)
	# compute targets
	results = [objective([x]) for x in inputs]
	# create a line plot of input vs result
	pyplot.plot(inputs, results)
	# define optimal input value
	x_optima = example['goal']  # 0.0
	print('x_optima:', x_optima)
	# draw a vertical line at the optimal input
	pyplot.axvline(x=x_optima, ls='--', color='red')
	# show the plot
	pyplot.title('Objective function')
	gm.save_figure(example['graph_file'])
	pyplot.show()

	# seed the pseudorandom number generator
	seed(1)
	# define range for input
	bounds = asarray([[-5.0, 5.0]])
	# define the total iterations
	n_iterations = example['n_iters']  # 100
	# define the maximum step size
	step_size = example['step_size']  # 0.1
	# initial temperature
	temp = 10
	# perform the simulated annealing search
	best, score, scores = simulated_annealing(objective, bounds, n_iterations, step_size, temp)
	print('Done!')
	print('f(%s) = %f' % (best, score))
	print('scores:\n', scores)
	# line plot of best scores
	pyplot.plot(scores, '.-')
	pyplot.xlabel('Improvement Number')
	pyplot.ylabel('Evaluation f(x)')
	pyplot.title('Scores plot')
	pyplot.show()

	inputs = np.arange(bounds[0, 0], bounds[0, 1], 0.1)
	# create a line plot of input vs result
	pyplot.plot(inputs, [objective([x]) for x in inputs], '--')
	# draw a vertical line at the optimal input
	pyplot.axvline(x=x_optima, ls='--', color='red')
	# plot the sample as black circles
	pyplot.plot(lsolutions, [objective(x) for x in lsolutions], 'o', color='black')
	pyplot.title('Tested solutions')
	gm.save_figure(example['path_file'])
	pyplot.show()
	print('\nTotal time:', time() - begin)
