"""
This is a version of Breadth-First Search (BFS)
for searching a position in a board

BFS expands nodes in order of their depth from the root,
generating one level of the tree at a time until a solution is found.
Maintain a FIFO queue of nodes, initially containing just the root,
and always removing the node at the head of the queue, expanding it,
and adding its children to the tail of the queue

NOTE: this implementation doesn't work with weighted graphs
"""
__author__ = "Andres Soto Villaverde"
__email__ = "andres.soto.v@gmail.com"

# import graph_module as gm
import board_module as bm
from graph_examples import example
from time import time
import networkx as nx
import numpy as np
from typing import List
from pprint import pprint


def get_adj1(c, n):
    """
    get admitted adjacent positions of c
    between 0 and n-1. 
    Return those possible indexes
    """    
    vals = [c]
    if c > 0:
        vals.append(c-1)
    if c < (n-1):
        vals.append(c+1)
    return vals


def get_adjacents(current):
    """
    get admitted adjacent positions of current 
    which is a 2D position. It also restrict 
    those positions which are blocked.
    Return a list with those admitted positions
    """
    (a, b) = current
    avals = get_adj1(a, nrows)
    bvals = get_adj1(b, ncols)
    ladjs = [(r,c) for r in avals for c in bvals
             if r != a and b != c]
    for tup in blocked:
        if tup in ladjs:
            ladjs.remove(tup)
    return ladjs


def board_breadth_first_search(start: int, goal: int) -> List:
    """
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
    # path = nx.DiGraph()
    path = []
    path_found = False # flag to indicate if goal was found
    while queue:  # while queue is non-empty
        # print('\nvisited:', visited)
        print('queue:', queue)
        # remove the head of the queue
        current = queue.pop(0)
        # add it to the list of visited nodes to prevent cycles
        visited.append(current)
        print('Visited so far: ', visited)
        # print('Current path: ', path.edges)
        print('current:', current)
        if current == goal: # goal was reached
            path_found = True # goal found
            print('Path found!')
            break # end the search
        # enqueue all unvisited neighbors of current
        # for neighbor in list(graph.adj[current]):
        for neighbor in get_adjacents(current):
            # print('neighbor= ', neighbor)
            if neighbor not in visited:
                # path.add_edge(neighbor, current)
                path.append((neighbor, current))
                queue.append(neighbor)
                # visited.append(neighbor)
    if path_found:
        rev = bm.get_path(start, goal, path)
        return rev
    else:
        return [] # path not found


if __name__ == '__main__':
    # Get example data
    # resp = example(11)  # simple board
    resp = example(12)  # with blocked
    # Define the number of rows and columns of the board
    nrows = resp['nrows']
    ncols = resp['ncols']
    # Define the start position for the search
    start = resp['start']
    # Define the search goal
    goal = resp['goal']
    print('number of rows:', nrows)
    print('number of columns:', ncols)
    print('start position:', start)
    print('goal:', goal)
    rgb = bm.set_colors(nrows, ncols)
    blocked = resp['blocked']
    print('blocked:', blocked)
    for (r, c) in blocked:
        rgb = bm.update_color(rgb, r, c, bm.red)
    print('rgb shape:', rgb.shape)
    bm.draw_board(nrows, ncols, rgb, resp['board_plot'])
    begin = time()
    path = board_breadth_first_search(start, goal)
    print('\nTotal time:', time() - begin)
    if len(path) > 0:
        print('Path from', start, 'to', goal, '=', path)
        bm.plot_path(rgb, nrows, ncols, path, resp['path_plot'])
    else:
        print('Goal ', goal, 'not found')




