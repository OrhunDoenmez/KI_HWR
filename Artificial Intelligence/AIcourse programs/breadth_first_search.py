"""
Breadth-First Search (BFS)

BFS expands nodes in order of their depth from the root,
generating one level of the tree at a time until a solution is found.
Maintain a FIFO queue of nodes, initially containing just the root,
and always removing the node at the head of the queue, expanding it,
and adding its children to the tail of the queue

NOTE: this implementation doesn't work with weighted graphs
"""
__author__ = "Andres Soto Villaverde"
__email__ = "andres.soto.v@gmail.com"

import graph_module as gm
from graph_examples import example

from time import time
import networkx as nx
from typing import List
from pprint import pprint

def breadth_first_search(graph: nx.DiGraph, start: int, goal: int) -> List:
    """
    :param graph: graph containing the nodes and edges
    :param start: where to start the search
    :param goal: expected end of the search
    :return: list of nodes visited to reach the goal
    """
    visited = []  # List for visited nodes.
    queue = [start]  # queue of nodes to be processed
                        # initialized with the start node
    print('\nvisited:', visited)
    print('queue:', queue)

    # Loop to visit each node
    path = nx.DiGraph()
    path_found = False # flag to indicate if goal was found
    while queue:  # while queue is non-empty
        # print('\nvisited:', visited)
        print('queue:', queue)
        # remove the head of the queue
        current = queue.pop(0)

        # add it to the list of visited nodes to prevent cycles
        visited.append(current)
        print('Visited so far: ', visited)
        print('Current path: ', path.edges)
        print('current:', current)
        if current == goal: # goal was reached
            path_found = True # goal found
            break # end the search
        # enqueue all unvisited neighbors of current
        for neighbor in list(graph.adj[current]):
            print('neighbor= ', neighbor)
            if neighbor not in visited:
                path.add_edge(neighbor, current)
                queue.append(neighbor)
                # visited.append(neighbor)
    if path_found:
        rev = gm.get_path(start, goal, path)
        gm.plot_path(G, rev, resp['path_file'])
        return rev # visited
    else:
        return [] # path not found


if __name__ == '__main__':
    # Get example data
    resp = example(1)
    # resp = example(2)
    # Create the graph
    pprint(resp['graph_dic'])
    G = gm.create_graph(resp['graph_dic'])
    # Plot the graph and save it
    gm.plot_graph(G, resp['graph_file'])  # 'bfs_rnd_g1.png'
    # Define the start and goal nodes
    start = resp['start'] # 3 # 2 # 5
    goal = resp['goal'] # 1
    print('start:', start, 'goal:', goal)
    begin = time()
    path = breadth_first_search(G, start, goal)
    print('\nTotal time:', time() - begin)
    if len(path) > 0:
        print('Path from', start, 'to', goal, '=', path)
    else:
        print('Goal ', goal, 'not found')




