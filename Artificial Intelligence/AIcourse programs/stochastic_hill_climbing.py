"""
Stochastic Hill Climbing in Python from Scratch
https://machinelearningmastery.com/stochastic-hill-climbing-in-python-from-scratch/
"""

from numpy import asarray
from numpy import arange
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed
from matplotlib import pyplot
from time import time
from typing import List
import graph_examples as ge
import graph_module as gm

"""
Let's define our objective function 
"""
def objective(x):
	# return x[0]**4 - 4 * x[0]**3 - 2*x[0]**2 + 5*x[0] + 9 # function 2
	return x[0]**2.0 # function1


# hill climbing local search algorithm
def hillclimbing(objective, bounds, n_iterations, step_size):
	# generate an initial point between any two given numbers A and B (A < B)
	#  B + ( B - A ) * RAND
	solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	print('solution:', solution) # [-2.78006829]
	# evaluate the initial point
	solution_eval = objective(solution)
	print('initial point evaluation:', solution_eval)
	# run the hill climb
	solutions = list()
	solutions.append(solution)
	for i in range(n_iterations):
		print('iteration=', i)
		# take a step
		candidate = solution + randn(len(bounds)) * step_size
		# evaluate candidate point
		candidte_eval = objective(candidate)
		# check if we should keep the new point
		if candidte_eval <= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidte_eval
			# keep track of solutions
			solutions.append(solution)
			# report progress on iteration i
			print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
	return [solution, solution_eval, solutions]


if __name__ == '__main__':
	example = ge.example(5)  # ex.6 func.2 # ex.5 func.1
	begin = time()
	# define range for input
	r_min, r_max = -5.0, 5.0 # function 1
	# sample input range uniformly at 0.1 increments
	inputs = arange(r_min, r_max, 0.1)
	# compute targets
	results = [objective([x]) for x in inputs]
	# create a line plot of input vs result
	pyplot.plot(inputs, results)
	# define optimal input value
	x_optima = example['goal']  # 0.0
	# draw a vertical line at the optimal input
	pyplot.axvline(x=x_optima, ls='--', color='red')
	# show the plot
	pyplot.title('Objective function')
	gm.save_figure(example['graph_file'])
	pyplot.show()

	"""
	Next, we can apply the hill climbing algorithm to the objective function.
	First, we will seed the pseudorandom number generator.
	"""
	# seed the pseudorandom number generator
	seed(5)
	"""
	Next, we can define the configuration of the search.
	In this case, we will search for 1,000 iterations of the algorithm and 
	use a step size of 0.1. 
	Given that we are using a Gaussian function for generating the step, 
	this means that about 99 percent of all steps taken will be within a 
	distance of (0.1 * 3) of a given point, e.g. three standard deviations.
	"""
	# define the total iterations
	n_iterations = example['n_iters']  # 1000
	# define the maximum step size
	step_size = example['step_size']  # 0.1
	# define range for input
	"""
	let's define the bounds on each input variable to the objective function. 
	The bounds will be a 2D array with one dimension for each input variable 
	that defines the minimum and maximum for the variable.
	"""
	bounds = asarray([[-5.0, 5.0]]) # function 1
	print('bounds:', bounds) # [[-5.  5.]]
	"""
	Next, we can perform the search and report the results.
	"""
	# perform the hill climbing search
	best, score, solutions = hillclimbing(objective, bounds, n_iterations, step_size)
	print('Done!')
	print('f(%s) = %f' % (best, score))
	inputs = arange(bounds[0,0], bounds[0,1], 0.1)
	# create a line plot of input vs result
	pyplot.plot(inputs, [objective([x]) for x in inputs], '--')
	# draw a vertical line at the optimal input
	pyplot.axvline(x=x_optima, ls='--', color='red')
	# plot the sample as black circles
	pyplot.plot(solutions, [objective(x) for x in solutions], 'o', color='black')
	pyplot.title('Tested solutions')
	gm.save_figure(example['path_file'])
	pyplot.show()
	print('\nTotal time:', time() - begin)
