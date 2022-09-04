"""
Depth First Search
DFS algorithm starts from the root node and 
explores as far as possible along each branch 
before backtracking, i.e., going back to an 
earlier point in a sequence. 

NOTE: it doesn't work with weighted graphs
"""
__author__ = "Andres Soto Villaverde"
__email__ = "andres.soto.v@gmail.com"

import networkx as nx
import graph_module as gm
from graph_examples import example
from time import time
from typing import List


def dfs(graph, start, target, path = [], visited = set()):
    path.append(start)
    visited.add(start)
    if start == target:
        rev = gm.get_path(resp['start'], resp['goal'], route)
        gm.plot_path(graph, rev, resp['path_file'])
        return rev
    adj_start = [(elem[1], elem[2]) for elem in graph.edges.data('w') if elem[0] == start]
    print('\nstart=',start)
    print('adj_start:', adj_start)
    for (neighbour, weight) in adj_start: # graph.m_adj_list[start]:
        if neighbour not in visited:
            route.add_edge(neighbour, start)
            result = dfs(graph, neighbour, target, path, visited)
            if result is not None:
                return result
    path.pop()
    return None


if __name__ == '__main__':
    # Get example data
    resp = example(1)
    # resp = example(2)
    # resp['graph_file'] = 'graph_dfs_ex2'
    # resp['path_file'] = 'path_dfs_ex2'
    # Create the graph
    graph = gm.create_graph(resp['graph_dic'])
    # Plot the graph and save it
    gm.plot_graph(graph, resp['graph_file'])
    # Define the start and goal nodes
    start = resp['start']  # 3
    goal = resp['goal']  # 1
    # # search from start node to goal node
    #     start = 0
    #     goal = 4 # 6 # 5
    print('start:', start, 'goal:', goal)
    begin = time()
    route = nx.DiGraph()
    tpath = dfs(graph, start, goal)
    print('\nTotal time:', time() - begin)
    if len(tpath) > 0:
        print('Path from', start, 'to', goal, '=', tpath)
    else:
        print('Goal ', goal, 'not found')
    print('\nDFS path:', tpath, 'from', start, 'to', goal)
    pass