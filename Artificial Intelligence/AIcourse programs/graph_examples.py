"""
This module contains several methods to obtain
the data corresponding to different examples.
It uses a switcher included in the method example
to execute the method that corresponds to the
indicated example.
In all cases, it returns a dictionary which contains
the graph dictionary, the graph, the files to save
the graph plot and the path plot, and the start and
goal nodes
NOTE: to create the new examples use graph_module
"""

import graph_module as gm


def rnd_graph1():
    """
    Define the data corresponding to the random graph 1:
    the graph dictionary, the graph, the files to save
    the graph plot and the path plot and the start and goal nodes
    :return: a dictionary with the data required for the example
    """
    resp = {}
    # Create the graph
    resp['graph_dic'] = gm.load_graph('rnd_graph1') # load dict with graph data
    resp['graph'] = gm.create_graph(resp['graph_dic'])
    # Define the file to save the graph plot
    resp['graph_file'] = 'bfs_rnd_g1.png'
    # Define the file to save the path plot
    resp['path_file'] = 'bfs_rnd_p1.png'
    # Define the start and goal nodes
    resp['start'] = 3  # 2 # 5
    resp['goal'] = 1
    print('start:', resp['start'], 'goal:', resp['goal'])
    return resp


def rnd_graph2():
    """
    Idem rnd_graph1, but the adjacencies of node 3 are inverted
    (the file to be load and the files to save the plots are changed)
    :return: a dictionary with the data required for the example
    """
    resp = {}
    # Create the graph
    resp['graph_dic'] = gm.load_graph('rnd_graph2')
    resp['graph'] = gm.create_graph(resp['graph_dic'])
    # Define the file to save the graph plot
    resp['graph_file'] = 'bfs_rnd_g2.png'
    # Define the file to save the path plot
    resp['path_file'] = 'bfs_rnd_p2.png'
    # Define the start and goal nodes
    resp['start'] = 3  # 2 # 5
    resp['goal'] = 1
    print('start:', resp['start'], 'goal:', resp['goal'])
    return resp


def dollar2G_graph():
    resp = {}
    # Create the graph
    # resp['graph_dic'] = gm.load_graph('rnd_graph2')
    # resp['graph'] = gm.create_graph(resp['graph_dic'])
    resp['graph'] = gm.create_graph_dollar2G()
    # Define the file to save the graph plot
    resp['graph_file'] = 'dollar2G_g1.png'
    # Define the file to save the path plot
    resp['path_file'] = 'dollar2G_p1.png'
    # Define the start and goal nodes
    resp['start'] = 0
    resp['goal'] = 4 # 6 # 5
    print('start:', resp['start'], 'goal:', resp['goal'])
    return resp


def hill_climbing1():
    """
    Define the data corresponding to the
    stochastic hill climbing example 1
    the objective function plot (graph_file),
    the tested solutions plot (path_file),
    the number of iterations (n_iters),
    the step size, and the optimal input value (goal)
    :return: a dictionary with the data required for the example
    """
    resp = {}
    # Define the file to save the objective function plot
    resp['graph_file'] = 'SHCobj_func1.png'
    # Define the file to save the tested solutions plot
    resp['path_file'] = 'SHCtested_sol1.png'
    # Define the number of iterations
    resp['n_iters'] = 100
    # Define the step size
    resp['step_size'] = 0.1
    # Define the optimal input value
    resp['goal'] = 0.0
    return resp


def hill_climbing2():
    """
    Define the data corresponding to the
	stochastic hill climbing example 2
    the objective function plot (graph_file),
    the tested solutions plot (path_file),
    the number of iterations (n_iters),
    the step size, and the optimal input value (goal)
    :return: a dictionary with the data required for the example
    """
    resp = {}
    # Define the file to save the Objective function plot
    resp['graph_file'] = 'SHCobj_func2.png'
    # Define the file to save the Tested solutions plot
    resp['path_file'] = 'SHCtested_sol2.png'
    # Define the number of iterations
    resp['n_iters'] = 100
    # Define the step size
    resp['step_size'] = 0.1
    # Define the optimal input value
    resp['goal'] = 3.2
    return resp


def simulated_annealing1():
    """
    Define the data corresponding to the simulated annealing example 1
    the objective function plot (graph_file),
    the tested solutions plot (path_file),
    the number of iterations (n_iters),
    the step size, and the optimal input value (goal)
    :return: a dictionary with the data required for the example
    """
    resp = {}
    # Define the file to save the objective function plot
    resp['graph_file'] = 'SAobj_func1.png'
    # Define the file to save the tested solutions plot
    resp['path_file'] = 'SAtested_sol1.png'
    # Define the number of iterations
    resp['n_iters'] = 100
    # Define the step size
    resp['step_size'] = 0.1
    # Define the optimal input value
    resp['goal'] = 0.0
    return resp


def simulated_annealing2():
    """
    Define the data corresponding to the simulated annealing example 2
    the objective function plot (graph_file),
    the tested solutions plot (path_file),
    the number of iterations (n_iters),
    the step size, and the optimal input value (goal)
    :return: a dictionary with the data required for the example
    """
    resp = {}
    # Define the file to save the Objective function plot
    resp['graph_file'] = 'SAobj_func2.png'
    # Define the file to save the Tested solutions plot
    resp['path_file'] = 'SAtested_sol2.png'
    # Define the number of iterations
    resp['n_iters'] = 100
    # Define the step size
    resp['step_size'] = 0.1
    # Define the optimal input value
    resp['goal'] = 3.2
    return resp


def genetic_algorithm_x2():
    """
    Define the data corresponding to the genetic algorithm example
    f(x) = x^2
    the objective function plot (graph_file),
    tested solutions plot (path_file), bounds,
    number of iterations (n_iters), number of bits,
    population size, and the crossover and mutation hyper-params
    :return: a dictionary with the data required for the example
    """
    resp = {}
    # Define the file to save the Objective function plot
    resp['graph_file'] = 'GAobj_func_x2.png'
    # Define the file to save the Tested solutions plot
    resp['path_file'] = 'GAtested_sol_x2.png'
    # define the bounds of each input variable.
    resp['bounds'] = [[-5.0, 5.0]]
    # define the total iterations
    resp['n_iter'] = 100
    # number of bits per input variable
    resp['n_bits'] = 16
    # define the population size
    resp['n_pop'] = 100
    # crossover rate
    resp['r_cross'] = 0.9
    # mutation rate
    resp['r_mut'] = 1.0 / (float(resp['n_bits']) * len(resp['bounds']))
    return resp


def genetic_algorithm_x4():
    """
    Define the data corresponding to the genetic algorithm example
    f(x) = x[0]**4 - 4 * x[0]**3 - 2*x[0]**2 + 5*x[0] + 9
    the objective function plot (graph_file),
    tested solutions plot (path_file), bounds,
    number of iterations (n_iters), number of bits,
    population size, and the crossover and mutation hyper-params
    :return: a dictionary with the data required for the example
    """
    resp = {}
    # Define the file to save the Objective function plot
    resp['graph_file'] = 'GAobj_func_x4.png'
    # Define the file to save the Tested solutions plot
    resp['path_file'] = 'GAtested_sol_x4.png'
    # define the bounds of each input variable.
    resp['bounds'] = [[-5.0, 5.0]]
    # define the total iterations
    resp['n_iter'] = 100
    # number of bits per input variable
    resp['n_bits'] = 16
    # define the population size
    resp['n_pop'] = 100
    # crossover rate
    resp['r_cross'] = 0.9
    # mutation rate
    resp['r_mut'] = 1.0 / (float(resp['n_bits']) * len(resp['bounds']))
    return resp


def board_example1():
    resp = {}
    # Define the number of rows and columns of the board
    resp['nrows'] = 5
    resp['ncols'] = 5
    # Define the start position for the search
    resp['start'] = (4, 4)
    # Define the search goal
    resp['goal'] = (0, 0)
    resp['blocked'] = []
    resp['board_plot'] = 'BBFSboard_file1.png'
    resp['path_plot'] = 'BBFSpath_file1.png'
    return resp


def board_example2():
    resp = {}
    # Define the number of rows and columns of the board
    resp['nrows'] = 5
    resp['ncols'] = 5
    # Define the start position for the search
    resp['start'] = (4, 4)
    # Define the search goal
    resp['goal'] = (0, 0)
    resp['blocked'] = [(2, 1), (2, 2)]
    resp['board_plot'] = 'BBFSboard_file2.png'
    resp['path_plot'] = 'BBFSpath_file2.png'
    return resp


def test1():
    """
    Define the data corresponding to a random graph
    used for testing the students.
    the graph dictionary, the graph, the files to save
    the graph plot and the path plot and the start and goal nodes
    :return: a dictionary with the data required for the example
    """
    resp = {}
    # Create the graph
    resp['graph_dic'] = {0: [2, 3, 4],
                         1: [0, 6],
                         2: [4, 6],
                         3: [2, 4, 5, 6],
                         4: [0, 1, 2, 3, 5, 6],
                         5: [0, 1, 3, 4],
                         6: [1, 2, 4, 5]}
    resp['graph'] = gm.create_graph(resp['graph_dic'])
    # Define the file to save the graph plot
    resp['graph_file'] = 'rnd_test_g1.png'
    # Define the file to save the path plot
    resp['path_file'] = 'rnd_test_p1.png'
    # Define the start and goal nodes
    resp['start'] = 0
    resp['goal'] = 1
    print('start:', resp['start'], 'goal:', resp['goal'])
    return resp



def example(num: int) -> object:
    switcher = {
        1: rnd_graph1,
        2: rnd_graph2,
        3: dollar2G_graph,
        4: d2G,
        5: hill_climbing1,
        6: hill_climbing2,
        7: simulated_annealing1,
        8: simulated_annealing2,
        9: genetic_algorithm_x2,
        10: genetic_algorithm_x4,
        11: board_example1,
        12: board_example2,
        13: test1,
    }
    # Get the function from switcher dictionary
    func = switcher.get(num, lambda: "Invalid month")
    # Execute the function
    return func()


def d2G():
    resp = {}
    # load dict with graph data
    resp['graph_dic'] = gm.load_graph('dollar2G')
    # Create the graph
    resp['graph'] = gm.create_weighted_graph(resp['graph_dic'])
    # Define the file to save the graph plot
    resp['graph_file'] = 'dollar2G_g1.png'
    # Define the file to save the path plot
    resp['path_file'] = 'dollar2G_p1.png'
    # Define the start and goal nodes
    resp['start'] = 0
    resp['goal'] = 6 # 4 # 5
    print('start:', resp['start'], 'goal:', resp['goal'])
    return resp


if __name__ == '__main__':
    #     example(1)
    #     example(2)

    pass


# def aco_graph1():
#     resp = {}
#     # Create the graph
#     # resp['graph_dic'] = gm.load_graph('rnd_graph2')
#     # resp['graph'] = gm.create_graph(resp['graph_dic'])
#     resp['graph'] = gm.create_aco_graph1()
#     # Define the file to save the graph plot
#     resp['graph_file'] = 'aco_graph1_g1.png'
#     # Define the file to save the path plot
#     resp['path_file'] = 'aco_graph1_p1.png'
#     # Define the start and goal nodes
#     resp['start'] = 0
#     resp['goal'] = 1
#     print('start:', resp['start'], 'goal:', resp['goal'])
#     return resp