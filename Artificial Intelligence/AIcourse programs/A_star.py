"""
A* (A-star)
"""
__author__ = "Andres Soto Villaverde"
__email__ = "andres.soto.v@gmail.com"

import networkx as nx
# import matplotlib.pyplot as plt
from operator import itemgetter
import numpy as np
from pprint import pprint
# import graph_astar as gastar
import graph_module as gm
from graph_examples import example
from time import time
from typing import List


def a_star(graph, start, goal):
    # gScore[n] is the cost of the cheapest path
    # from start to n currently known.
    # gScore[start] = 0
    graph.nodes[start]['gScore'] = 0
    # fScore[n] := gScore[n] + hScore(n).
    # fScore[n] represents our current best guess as to
    # how cheap a path  could be from start to finish if
    # it goes through n.
    graph.nodes[start]['fScore'] = graph.nodes[start]['gScore'] + \
                                   graph.nodes[start]['hScore']
    visited = []  # List for already visited nodes.
    # queue of nodes to be processed
    queue = [(start, graph.nodes[start]['fScore'])]
    # initialized with the start node
    # flag to indicate if goal was found
    # path_found = False
    # cost0 = 0
    # Loop to visit each node
    path = nx.DiGraph()
    while queue:  # while queue is non-empty
        # queue is sorted according to fScore
        queue.sort(key=itemgetter(1))
        print('\nqueue: ', queue)
        # remove the head of the queue
        # current is  the node having the lowest fScore[]
        (current, fScore) = queue.pop(0)
        # add it to the list of visited nodes to prevent cycles
        visited.append(current)
        print('Visited so far: ', visited)
        # print('Current path: ', path.edges.data('w'))
        if current == goal:  # goal was reached
            rev = gm.get_path(start, goal, path)
            gm.plot_weighted_path(graph, rev, resp['path_file'])
            return rev  # [rev, fScore]
            # rev = get_path(start, goal, path)
            #         return [rev, fScore]
        print('current= ', current)
        # enqueue all unvisited neighbors of current sorted by weight
        for neighbor in list(graph.adj[current]):
            print('neighbor= ', neighbor)
            # graph[current][neighbor]['w'] is the weight of the edge from
            # current to neighbor
            # auxScore is the distance from start to the neighbor
            # through current
            auxScore = graph.nodes[current]['gScore'] + \
                       graph[current][neighbor]['w']
            if auxScore < graph.nodes[neighbor]['gScore']:
                # path to neighbor is better than previous one.
                graph.nodes[neighbor]['gScore'] = auxScore
                graph.nodes[neighbor]['fScore'] = auxScore + \
                                                  graph.nodes[neighbor]['hScore']
                path.add_edge(neighbor, current, w=graph[current][neighbor]['w'])  # graph.nodes[neighbor]['fScore'])
                if neighbor not in visited:
                    queue.append((neighbor, graph.nodes[neighbor]['fScore']))
    #     if path_found:
    #         rev = get_path(start, goal, path)
    #         return [rev, fScore]
    #     else:
    #         return [] # path not found
    return []  # path not found


# def get_path(start, goal, path):
#     print('\nAuxiliar Graph')
#     ledges = list(path.edges.data('w'))
#     ledges = ledges[::-1]
#     for edge in ledges:
#         print(edge)
#     print()
#     edge = ledges[0]  # first node contains the goal
#     st3 = [edge]
#     st1 = [goal]
#     # print('st1: ', st1)
#     succ = edge[1]
#     while succ != start:
#         st1.append(succ)
#         # print('st1: ', st1)
#         edge = next(x for x in ledges if x[0] == succ)
#         st3.append(edge)
#         succ = edge[1]
#
#     st1.append(succ)
#     # print('st1: ', st1)
#     rev = st1[::-1]
#     # print('path: ', rev)
#     return rev


if __name__ == '__main__':
    gScore = np.inf  # []
    hScore = 1
    fScore = 0
    # Get example data
    resp = example(3)
    # Create the graph
    graph = gm.create_graph_dollar2G()
    nx.set_node_attributes(graph, gScore, 'gScore')
    nx.set_node_attributes(graph, hScore, 'hScore')
    nx.set_node_attributes(graph, fScore, 'fScore')
    # Plot the graph and save it
    resp['graph_file'] = 'astartg.png'
    resp['path_file'] = 'astartp.png'
    gm.plot_weighted_graph(graph, resp['graph_file'])
    # Define the start and goal nodes
    start = resp['start']  # 0
    goal = resp['goal']  # 4 # 6 # 5
    print('start:', start, 'goal:', goal)
    begin = time()
    # graph_file = 'astart.png'
    #     gm.plot_graph(graph, graph_file)
    #     # search from start node to goal node
    #     start = 0
    #     goal = 4 # 6 # 5 # 4
    path = a_star(graph, start, goal)
    if len(path) > 0:
        # [path, cost] = resp
        print('Visited ', start, 'to ', goal)  # , 'cost= ', cost)
        print('Path:\n', path)
    else:
        print('Goal ', goal, 'not found')
