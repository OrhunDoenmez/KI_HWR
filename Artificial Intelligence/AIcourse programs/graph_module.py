"""
Auxiliar module with several methods useful while
searching for a path from a start to a goal node
* Create a list with n nodes and connect those
    nodes at random using a dictionary to keep the
    edges between the nodes
* Create a networkx graph corresponding to the nodes
    and edges described by the dictionary gdic
* Plot the graph G balancing the distances between
    the node positions and save the figure in a file
* Plot the path from start to goal previously obtained by
    the search algorithm. The path appears in red.
* Extract the shortest path from start to goal obtained
    after searching a graph from the start to the goal.
* Methods to save the dictionary graph in a pickle file
    and to load it
"""

__author__ = "Andres Soto Villaverde"
__email__ = "andres.soto.v@gmail.com"

import networkx as nx
import matplotlib.pyplot as plt
import random as rnd
import numpy as np
from pprint import pprint
import pickle
from typing import List


import sys
from pathlib import Path
import configuration as config
import os

sys.path.append(str(Path('.').absolute().parent))

seed = 2022
rnd.seed(seed)
np.random.seed(seed)


def rnd_graph(n: int) -> dict:
    """
    Create a list with n nodes and connect those
    nodes at random using a dictionary to keep the
    edges between the nodes
    :param n: number of nodes
    :return: dictionary with n nodes and the connections
            between the nodes
    """
    # initialize the graph
    lnodes = list(range(n))
    gdic = {item: [] for item in lnodes}
    for i in lnodes:
        for j in lnodes:
            if i != j and rnd.random() < 0.3:
                gdic[i].append(j)
                print('Edge from', i, 'to', j)
    pprint(gdic)
    return gdic


def rnd_weighted_graph(n: int) -> dict:
    """
    Create a list with n nodes and connect those
    nodes at random using a dictionary to keep the
    edges between the nodes
    :param n: number of nodes
    :return: dictionary with n nodes and the connections
            between the nodes
    """
    # initialize the graph
    lnodes = list(range(n))
    gdic = {item: [] for item in lnodes}
    for i in lnodes:
        for j in lnodes:
            if i != j and rnd.random() < 0.5:
                k = rnd.randint(1, 10)
                gdic[i].append((j, k))
                print('Edge from', i, 'to', j, 'weight', k)
    pprint(gdic)
    return gdic


def create_weighted_graph(gdic: dict) -> nx.DiGraph:
    # initialize the graph
    G = nx.DiGraph()
    lnodes = gdic.keys()
    G.add_nodes_from(lnodes)
    for node in lnodes:
        # lneighbors = gdic[node]
        for (neighbor, weight) in gdic[node]:
            G.add_edge(node, neighbor, w=weight)
    # Graph summary
    print('number_of_nodes: ', G.number_of_nodes())
    print('Graph nodes: ', list(G.nodes))
    print('number_of_edges:', G.number_of_edges())
    print('Graph edge weights:')
    for edge in G.edges.data('w'):
        print(edge)
    print()
    return G


def create_graph(gdic: dict) -> nx.DiGraph:
    """
    Create a networkx graph corresponding to the nodes
    and edges described by the dictionary gdic
    :param gdic: dictionary where the keys define the
                graph nodes and list of values associated
                with each key indicated the connections
                between the nodes
    :return: the graph created
    """
    # initialize the graph
    G = nx.DiGraph()
    # lnodes = list(range(6))
    lnodes = gdic.keys()
    G.add_nodes_from(lnodes)
    for i in lnodes:
        for j in gdic[i]:
            G.add_edge(i, j)

    print('number_of_nodes:', G.number_of_nodes())
    print('Graph nodes:', list(G.nodes))
    print('number_of_edges:', G.number_of_edges())
    print('Node\tAdjacencies')
    for node in list(G.nodes):
        print(node, '\t', list(G.adj[node].keys()))
    return G


def create_graph_dollar2G():
    """
    idem to method create_graph in graphs_astar

    graph $(dollar) to G(goal)
    Example appearing in
    Uniform-Cost Search (Dijkstra for large Graphs)
    https://www.geeksforgeeks.org/uniform-cost-search-dijkstra-for-large-graphs/

    :return: Directed weighted graph with 7 nodes
    """
    # initialize the graph
    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5, 6])
    G.add_edge(0, 1, w=2)
    G.add_edge(0, 3, w=5)
    G.add_edge(1, 6, w=1)
    G.add_edge(2, 1, w=4)
    G.add_edge(3, 1, w=5)
    G.add_edge(3, 4, w=2)
    G.add_edge(3, 6, w=6)
    G.add_edge(4, 2, w=4)
    G.add_edge(4, 5, w=3)
    G.add_edge(5, 2, w=6)
    G.add_edge(5, 6, w=3)
    G.add_edge(6, 4, w=7)
    # Graph summary
    print('number_of_nodes: ', G.number_of_nodes())
    print('Graph nodes: ', list(G.nodes))
    print('number_of_edges:', G.number_of_edges())
    print('Graph edge weights:')
    for edge in G.edges.data('w'):
        print(edge)
    print()

    return G


def create_aco_graph1():
    # initialize the graph
    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2])
    G.add_edge(0, 1, length=1, tau=0.5)
    G.add_edge(0, 2, length=1, tau=0.5)
    G.add_edge(1, 2, length=1, tau=0.5)
    # Graph summary
    print('number_of_nodes: ', G.number_of_nodes())
    print('Graph nodes: ', list(G.nodes))
    print('number_of_edges:', G.number_of_edges())
    #     print('Graph edge length:')
    #     for edge in G.edges.data('length'):
    #         print(edge)
    #     print('Graph edge tau (pheromone value):')
    #     for edge in G.edges.data('tau'):
    #         print(edge)
    print('Edges with their labels')
    data = {}
    for n1, n2, d in G.edges.data():
        data[(n1, n2)] = d
    pprint(data)

    print()

    return G


def plot_graph(G: nx.DiGraph, gfile: str) -> None:
    """
    plot the graph G balancing the distances between
    the node positions and save the figure in a file
    :param G: graph
    :param gfile: name of a file to save the figure
    :return: None
    """
    # spring_layout forces a representation of the graph
    # balancing the distances between the node positions
    pos = nx.spring_layout(G, seed)  # seed for reproducibility
    nx.draw(G, pos, with_labels=True, font_weight='bold')
    # plt.savefig(gfile)
    save_figure(gfile)
    plt.show()
    plt.close()


def plot_weighted_graph(G, gfile):
    """
    plot the graph G balancing the distances between
    the node positions and save the figure in a file
    :param G: graph
    :param gfile: name of a file to save the figure
    :return: None
    """
    # plot the graph
    pos = nx.spring_layout(G, seed=seed)  # seed for reproducibility
    nx.draw(G, pos, with_labels=True, font_weight='bold')
    # edge_labels = nx.get_edge_attributes(G, "w")
    # nx.draw_networkx_edge_labels(G, pos, edge_labels)
    data = {}
    for n1, n2, d in G.edges.data():
        data[(n1, n2)] = d
    nx.draw_networkx_edge_labels(G, pos, data)
    # plt.savefig(gfile)
    save_figure(gfile)
    plt.show()


def get_path(start: int, goal: int, path: nx.DiGraph) -> List:
    """
    Extract the shortest path from start to goal obtained
    after searching a graph from the start to the goal.
    :param start: start node of the search
    :param goal: goal node of the search
    :param path: list of nodes visited while searching
                from start to goal
    :return: path from start to goal
    """
    # ledges: list of graph edges
    ledges = list(path.edges.data('w'))
    print('\n get_path list of edges\n')
    print(*ledges, sep='\n')
    # Reverse list of edges
    ledges = ledges[::-1]
    print('\nAuxiliar Path Graph')
    print('From, To, Weight')
    print(*ledges, sep='\n')
    # get the indexes of the edges which contains goal at first position
    lind = [ind for ind, edge in enumerate(ledges) if edge[0] == goal]

    print('\n Index list:', lind)

    # pt point to the index of the first edge that contains goal
    # as the first element, i.e., at first position
    pt = lind[0]
    print()
    edge = ledges[pt]  # this is the first edge that contains the goal
    print('1st edge:', edge) # (1, 2, None)
    # create a list to contain the nodes from goal to start
    rev_path = [goal]
    print('BEFORE reverse path:', rev_path) # [1]
    succ = edge[1]
    print('succ of goal:', succ) # 2
    print('start:', start)
    while succ != start: # start = 3
        rev_path.append(succ)
        print('reverse path:', rev_path)
        # The next() function returns the next item in an iterator.
        edge = next(x for x in ledges if x[0] == succ)
        print('succ:', succ)
        succ = edge[1]

    rev_path.append(succ)
    print('reverse path: ', rev_path)
    rev = rev_path[::-1]
    # print('path: ', rev)
    return rev


def test_rnd1() -> None:
    gdic = rnd_graph(6)
    graph_file = 'test.png'
    G = create_graph(gdic)
    plot_graph(G, graph_file)


def test_plot_path():
    gdic1 = rnd_graph(6)
    graph_file1 = 'test1.png'
    G1 = create_graph(gdic1)
    plot_path(G1, [3, 0, 4, 1], graph_file1)


def save_graph(gdic: dict, fname: str) -> None:
    """
    Save the dictionary graph in a pickle file
    :param gdic: dictionary with the graph nodes and
        their corresponding adjacent nodes
    :param fname: name of the file to save the dictionary
    :return: None
    """
    data_folder = Path(config.paths['data_path'])
    fname = fname + '.pkl'
    fpath = data_folder / fname
    pickle.dump(gdic, open(fpath, "wb"))


def load_graph(fname: str) -> dict:
    """
    Load the dictionary graph from a pickle file
    :param fname: name of the file containing save the dictionary
    :return: graph dictionary
    """
    data_folder = Path(config.paths['data_path'])
    fname = fname + '.pkl'
    fpath = data_folder / fname
    return pickle.load(open(fpath, "rb"))


def plot_path(G: nx.DiGraph, path: List, gfile: str) -> None:
    """
    Plot the path from start to goal previously obtained by
    the search algorithm. The path appears in red.
    :param G: networkx DiGraph with the whole graph searched
    :param path: list of path nodes from start to goal
    :param gfile: file name to save the figure in directory results
    :return: None
    """
    pos = nx.spring_layout(G, seed)  # seed for reproducibility
    nx.draw_networkx(G, pos, node_color='b',
                     with_labels=True, font_weight='bold')
    nx.draw_networkx(G, pos, nodelist=path, node_color='r',
                     with_labels=True, font_weight='bold')
    ltup = []
    for i in range(len(path) - 1):
        ltup.append((path[i], path[i + 1]))
    nx.draw_networkx_edges(G, pos, edgelist=ltup, arrows=True,
                           arrowsize=14, edge_color='r')
    save_figure(gfile)
    plt.show()
    plt.close()


def plot_weighted_path(G: nx.DiGraph, path: List, gfile: str) -> None:
    """
    Plot the path from start to goal previously obtained by
    the search algorithm. The path appears in red.
    :param G: networkx DiGraph with the whole graph searched
    :param path: list of path nodes from start to goal
    :param gfile: file name to save the figure in directory results
    :return: None
    """
    pos = nx.spring_layout(G, seed)  # seed for reproducibility
    nx.draw_networkx(G, pos, node_color='b',
                     with_labels=True, font_weight='bold')
    nx.draw_networkx(G, pos, nodelist=path, node_color='r',
                     with_labels=True, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, "w")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    ltup = []
    for i in range(len(path) - 1):
        ltup.append((path[i], path[i + 1]))
    nx.draw_networkx_edges(G, pos, edgelist=ltup, arrows=True,
                           arrowsize=14, edge_color='r')
    save_figure(gfile)
    plt.show()
    plt.close()


def save_figure(fig_file) -> None:
    """
    Method to change the directory to be able to save the figure
    in the directory results
    :param fig_file: name of a file to save the figure
    :return: None
    """
    # print('save_figure')
    path1 = os.getcwd()
    path2 = path1 + '\\results'
    os.chdir(path2)
    # print('current dir', path2)
    plt.savefig(fig_file)
    os.chdir(path1)


if __name__ == '__main__':
    # dic = load_graph('rnd_graph2')
    # dic = rnd_graph(7)
    #     pprint(dic)
    #     g = create_graph(dic)
    #     plot_graph(g, 'graph_example3')

    # graph = create_aco_graph1()
    #     plot_weighted_graph(graph, 'aco_graph1.png')  # , ['length', 'tau'])

    # dicg = load_graph('dollar2G')
    # pprint(dicg)
    dicg = rnd_weighted_graph(5)
    pprint(dicg)
    graph = create_weighted_graph(dicg)
    plot_weighted_graph(graph, 'rnd_weighted_graph5.png')
    # plot_weighted_graph(graph, 'dollar2G.png')

    # graph = create_graph(dicg_Dollar2G.dicg)
    # save_graph(dicg_Dollar2G.dicg, 'dollar2G')

    pass

