"""
Uniform Cost Search
"""
__author__ = "Andres Soto Villaverde"
__email__ = "andres.soto.v@gmail.com"

import networkx as nx
# import matplotlib.pyplot as plt
from operator import itemgetter
# import graph_dollar2G as dollar2G
import graph_module as gm
from graph_examples import example
from time import time
from typing import List

def uniform_cost_search(graph, start, goal):
    visited = []  # List for visited nodes.
    queue = [(start, 0)]  # queue of nodes to be processed
                    # initialized with the start node
                    # and its cost

    # Loop to visit each node
    path_found = False # flag to indicate if goal was found
    cost0 = 0
    path = nx.DiGraph()
    while queue:  # while queue is non-empty
        queue.sort( key=itemgetter(1))
        print('\nqueue: ', queue)
        # remove the head of the queue
        (current, cost0) = queue.pop(0)
        # add it to the list of visited nodes to prevent cycles
        visited.append(current)
        print('Visited so far: ', visited)
        print('Current path: ', path.edges.data('w'))
        if current == goal: # goal was reached
            path_found = True # goal found
            break # end the search
        print('current= ', current)
        # enqueue all unvisited neighbors of current sorted by weight
        for neighbor in list(graph.adj[current]):
            print('neighbor= ', neighbor)
            if neighbor not in visited:
                cost1 = graph[current][neighbor]['w']
                path.add_edge(neighbor, current, w=cost1)
                # print('Update cost\ncurrent:',current,'cost=',cost0)
                # print('neighbor:', neighbor, 'cost=',cost1, 'updated cost=', cost0 + cost1)
                queue.append((neighbor, cost0 + cost1))
    if path_found:
        rev = gm.get_path(start, goal, path)
        gm.plot_weighted_path(graph, rev, resp['path_file'])
        return [rev, cost0]
    else:
        return [] # path not found



if __name__ == '__main__':
    # Get example data
    resp = example(3)
    # Create the graph
    graph = gm.create_graph_dollar2G()
    # Plot the graph and save it
    gm.plot_weighted_graph(graph, resp['graph_file'])
    # Define the start and goal nodes
    start = resp['start'] # 0
    goal = resp['goal'] # 4 # 6 # 5
    print('start:', start, 'goal:', goal)
    begin = time()
    resp = uniform_cost_search(graph, start, goal)
    print('\nTotal time:', time() - begin)
    if len(resp) > 0:
        [path, cost] = resp
        print('Visited ', start, 'to ', goal, 'cost= ', cost)
        print('Path:\n', path)
    else:
        print('Goal ', goal, 'not found')


