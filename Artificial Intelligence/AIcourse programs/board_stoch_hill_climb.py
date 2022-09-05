"""
Version of Stochastic Hill Climbing in Python from Scratch
to work with a chess board
https://machinelearningmastery.com/stochastic-hill-climbing-in-python-from-scratch/
The Manhattan-Distance between two squares is determined by
the minimal number of orthogonal King moves between these squares
on the empty board
	dist = |x1 - x2| + |y1 - y2|
Observe that to move diagonal cost double
"""

from scipy.spatial.distance import cityblock
from numpy import asarray, arange
from numpy.random import randn, rand # , seed
import random as rnd
import numpy as np
import board_module as bm
from time import time
from typing import List


seed = 2022
rnd.seed(seed)
np.random.seed(seed)


"""
Let's define our objective function 
Our goal is to get to some cell on the chess board
so we have to calculate the distance from the 
current position to the goal cell.
"""
def objective(cell):
	"""
	Calculates the Manhattan distance from the current cell
	to the goal cell
	:param cell: tuple with current cell indexes
	:return: distance to the goal
	"""
	return cityblock(cell, goal)


def new_candidate(sol):
	(a, b) = sol
	a1 = a + rnd.randint(-1, 1)
	if not(0 <= a1 and a1 <= (nrows - 1)): # it's out of board
		a1 = a # keep the same
	b1 = b + rnd.randint(-1, 1)
	if not(0 <= b1 and b1 <= (ncols - 1)): # it's out of board
		b1 = b # keep the same
	blk = [tup for tup in blocked if tup[0] == a1 and tup[1] == b1]
	if len(blk) > 0: # this cell is blocked
		a1, b1 = sol # don't move, keep the same
	return (a1, b1)


# hill climbing local search algorithm
def hillclimbing(objective, bounds, n_iterations, step_size):
	# generate an initial point between any two given numbers A and B (A < B)
	#  B + ( B - A ) * RAND
	# solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	solution = (rnd.randint(0, nrows), rnd.randint(0, ncols))
	print('Initial solution:', solution) # (4, 2)
	# evaluate the initial point
	solution_eval = objective(solution)
	print('Initial solution estimate:', solution_eval)  # 6
	# run the hill climb
	solutions = list()
	solutions.append(solution)
	for i in range(n_iterations):
		print('iteration=', i)
		# take a step
		# candidate = solution + randn(len(bounds)) * step_size
		candidate = new_candidate(solution)
		print('candidate:', candidate)
		# evaluate candidate point
		candidate_eval = objective(candidate)
		print('Candidate evaluation:', candidate_eval)
		print('candidate evaluation:', candidate_eval)
		# check if we should keep the new point
		if candidate_eval <= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidate_eval
			# keep track of solutions
			solutions.append(solution)
			# report progress on iteration i
			print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
			print('Goal:', goal)
			if candidate == goal:
				print('Goal found!')
				return [solution, solution_eval, solutions]
	return [solution, solution_eval, solutions]


if __name__ == '__main__':
	begin = time()
	nrows = 5
	ncols = 5
	goal = (0, 0) # (3, 4) #
	print('number of rows:', nrows)
	print('number of columns:', ncols)
	print('goal:', goal)
	lrows = list(range(nrows))
	lcols = list(range(ncols))
	rgb = bm.set_colors(nrows, ncols)
	# Paint in red the blocked cells of the board
	blocked = [(2, 1), (2, 2)]
	for (r, c) in blocked:
		rgb = bm.update_color(rgb, r, c, bm.red)
	bm.draw_board(nrows, ncols, rgb, 'SHCboard.png')
	"""
	Next, we can define the configuration of the search.
	In this case, we will search for nrows*ncols iterations (i.e., the 
	total number of cells in the board) and use a step size of 1 (just
	to move to the next ortogonal cell)
	"""
	# define the total iterations
	# n_iterations = nrows * ncols
	n_iterations = 100
	# define the maximum step size
	step_size = 1
	"""
	let's define the bounds on each input variable to the objective function.
	The bounds will be a 2D array with one dimension for each input variable
	that defines the minimum and maximum for the variable.
	"""
	bounds = asarray([[0.0, (nrows - 1)],
					  [0.0, (ncols - 1)]
					  ])
	print('bounds:', bounds) # [[0. 4.] [0. 4.]]
	# perform the hill climbing search
	best, score, solutions = hillclimbing(objective, bounds,
										  n_iterations, step_size)
	print('Done!')
	print('f(%s) = %f' % (best, score))
	print('solutions:\n', solutions)
	rgb1 = 0
	# Paint in green the cells in the path
	for (r, c) in solutions:
		rgb1 = bm.update_color(rgb, r, c, bm.green)
	bm.draw_board(nrows, ncols, rgb1, 'SHCboard_sol.png')
	print('\nTotal time:', time() - begin)
	pass




