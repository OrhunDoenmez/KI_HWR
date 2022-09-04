"""
This is a version of Depth First Search (DFS)
for searching a position in a board
Depth First Search
DFS algorithm starts from the root node and
explores as far as possible along each branch
before backtracking, i.e., going back to an
earlier point in a sequence.

NOTE: it doesn't work with weighted graphs
"""
__author__ = "Andres Soto Villaverde"
__email__ = "andres.soto.v@gmail.com"

# import networkx as nx
# import graph_module as gm
import board_module as bm
from graph_examples import example
from time import time
from typing import List


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


def board_dfs(start, target, path = [], visited = set()):
    print('BFS start:', start, 'goal:', target)
    path.append(start)
    visited.add(start)
    if start == target:
        rev = bm.get_path(resp['start'], resp['goal'], route)
        # bm.plot_path(graph, rev, resp['path_file'])
        return rev
    # adj_start = [(elem[1], elem[2]) for elem in graph.edges.data('w') if elem[0] == start]
    adj_start = get_adjacents(start)
    print('\nstart=',start)
    print('adj_start:', adj_start)
    for neighbour in adj_start:
        if neighbour not in visited:
            route.append((neighbour, start))
            result = board_dfs(neighbour, target, path, visited)
            if result is not None:
                return result
    path.pop()
    return None


if __name__ == '__main__':
    # Get example data
    # resp = example(11)  # simple board
    #     resp['graph_file'] = 'board_BDFS_ex1'
    #     resp['path_file'] = 'path_BDFS_ex1'
    resp = example(12)  # with blocked
    resp['graph_file'] = 'board_BDFS_ex2'
    resp['path_file'] = 'path_BDFS_ex2'
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



    # # resp['graph_file'] = 'graph_dfs_ex2'
    #     # resp['path_file'] = 'path_dfs_ex2'
    #     # Create the graph
    #     graph = gm.create_graph(resp['graph_dic'])
    #     # Plot the graph and save it
    #     gm.plot_graph(graph, resp['graph_file'])
    #     # Define the start and goal nodes
    #     start = resp['start']  # 3
    #     goal = resp['goal']  # 1
    #     # # search from start node to goal node
    #     #     start = 0
    #     #     goal = 4 # 6 # 5
    #     print('start:', start, 'goal:', goal)
    #     begin = time()
    # route = nx.DiGraph()
    route = []
    tpath = board_dfs(start, goal)
    print('\nTotal time:', time() - begin)
    if len(tpath) > 0:
        print('Path from', start, 'to', goal, '=', tpath)
        bm.plot_path(rgb, nrows, ncols, tpath, resp['path_file'])
    else:
        print('Goal ', goal, 'not found')
    # print('\nDFS path:', tpath, 'from', start, 'to', goal)
    pass