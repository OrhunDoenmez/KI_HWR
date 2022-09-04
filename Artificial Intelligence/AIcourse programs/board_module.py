"""
This program allows to:
1) create a graph corresponding to a board
    with n rows and m columns assuming that
    each cell is connected with their neighbors
2) Draw the board with nrows x ncols with the RGB colors
3) Set the colors of the chess board in white and black
    alternating the positions
4) Update the color of board position rgb[row,col] = new_color
"""

__author__ = "Andres Soto Villaverde"
__email__ = "andres.soto.v@gmail.com"

import networkx as nx
import matplotlib.pyplot as plt
import random as rnd
import numpy as np
from pprint import pprint
from typing import List
from scipy.spatial.distance import cityblock
import graph_module as gm

seed = 2022
rnd.seed(seed)
np.random.seed(seed)
# Set colors using RGB
red = [255, 0, 0]
green = [0, 255, 0]
blue = [0, 0, 255]


def board2graph(n: int, m: int):
    """
    create a graph corresponding to a board
    with n rows and m columns assuming that
    each cell is connected with their neighbors
    :param n: number of rows
    :param m: number of columns
    :return:
    """
    gdic = {(i, j): [] for i in range(n) for j in range(m)}
    # cells not at corner
    for i in range(1, n-1):
        for j in range(1, m-1):
            gdic[(i,j)].append((i-1,j))
            gdic[(i,j)].append((i+1,j))
            gdic[(i,j)].append((i,j-1))
            gdic[(i,j)].append((i,j+1))

    # first row cells
    i = 0
    for j in range(1, m - 1):
        gdic[(i,j)].append((i+1,j))
        gdic[(i,j)].append((i,j-1))
        gdic[(i,j)].append((i,j+1))

    # last row cells
    i = n-1
    for j in range(1, m - 1):
        gdic[(i, j)].append((i - 1, j))
        gdic[(i, j)].append((i, j - 1))
        gdic[(i, j)].append((i, j + 1))

    # first column cells
    j = 0
    for i in range(1, n - 1):
        gdic[(i,j)].append((i-1,j))
        gdic[(i,j)].append((i+1,j))
        gdic[(i,j)].append((i,j+1))

    # last column cells
    j = m-1
    for i in range(1, n - 1):
        gdic[(i,j)].append((i-1,j))
        gdic[(i,j)].append((i+1,j))
        gdic[(i,j)].append((i,j-1))

    # upper left corner
    gdic[0, 0].append((0, 1))
    gdic[0, 0].append((1, 0))
    # upper right corner
    gdic[0, m-1].append((0, m-2))
    gdic[0, m-1].append((1, m-1))
    # lower left corner
    gdic[n-1, 0].append((n-1, 1))
    gdic[n-1, 0].append((n-2, 0))
    # lower right corner
    gdic[n-1, m-1].append((n-1, m-2))
    gdic[n-1, m-1].append((n-2, m-1))

    pprint(gdic)
    return gdic


def draw_board(nrows: int, ncols: int, rgb: np.array, bfile: str) -> None:
    """
    Draw the rgb board with nrows x ncols with the colors
    Matplotlib.pyplot.matshow() in Python - GeeksforGeeks
    https://www.geeksforgeeks.org/matplotlib-pyplot-matshow-in-python/
    :param nrows: number of rows
    :param ncols: number of columns
    :param rgb: ndarray with the colors of the cells
    :return: None
    """
    # print('\ndraw_board\n')
    figure = plt.figure()
    axes = figure.add_subplot(111)
    # using the matshow() function
    axes.matshow(rgb, interpolation='nearest')
    rlabels = [str(x) for x in range(nrows)]
    clabels = [str(x) for x in range(ncols)]
    # UserWarning_ FixedFormatter should only be used together with FixedLocator
    # Solution: use xticks in spite of set_xticklabels
    axes.set_xticklabels([''] + clabels)
    axes.set_yticklabels([''] + rlabels)
    # plt.savefig('geeks_board.png')
    gm.save_figure(bfile)
    plt.show()


def set_colors(nrows: int, ncols: int) -> np.array:
    """
    Set the colors of the chess board in white and black
    alternating the positions
    :param nrows: number of board rows
    :param ncols: number of board columns
    :return: colored board
    """
    # print('\nset_colors\n')
    # To calculate the alternate position for coloring,
    # the outer function brings a table (list of lists)
    # with 0 and 1 indicating which cell is colored
    # black or white
    z1 = np.add.outer(range(nrows), range(ncols)) % 2
    # rgb is a NxNx3 matrix of zeros
    rgb = np.zeros((nrows, ncols, 3))
    # Where we set the RGB for each pixel
    rgb[z1 > 0.5] = [1, 1, 1]
    rgb[z1 < 0.5] = [0, 0, 0]
    # print('RGB set:\n', rgb)
    return rgb


def update_color(rgb: np.array, row: int, col: int, new_color: List) -> np.array:
    """
    Update the color of board rgb[row,col] = new_color
    :param rgb: table with the colors of each board position
    :param row: row index
    :param col: column index
    :param new_color: new color to assign to the indicated element
    :return: updated board rgb
    """
    # print('\nupdate_color\n')
    #     print('Cell received:\n', rgb[row, col])
    rgb[row][col] = np.asarray(new_color)
    # print('Cell updated:\n', rgb[row, col])
    return rgb


def get_path(start: int, goal: int, path: List) -> List:
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
    # ledges = list(path.edges.data('w'))
    ledges = list(path)
    # print('\n get_path list of edges\n')
    # print(*ledges, sep='\n')
    # Reverse list of edges
    ledges = ledges[::-1]
    # print('\nAuxiliar Path Graph')
    # print('From, To, Weight')
    # print(*ledges, sep='\n')
    # get the indexes of the edges which contains goal at first position
    lind = [ind for ind, edge in enumerate(ledges) if edge[0] == goal]

    # print('\n Index list:', lind)

    # pt point to the index of the first edge that contains goal
    # as the first element, i.e., at first position
    pt = lind[0]
    print()
    edge = ledges[pt]  # this is the first edge that contains the goal
    # print('1st edge:', edge) # (1, 2, None)
    # create a list to contain the nodes from goal to start
    rev_path = [goal]
    # print('BEFORE reverse path:', rev_path) # [1]
    succ = edge[1]
    # print('succ of goal:', succ) # 2
    print('start:', start)
    while succ != start: # start = 3
        rev_path.append(succ)
        # print('reverse path:', rev_path)
        # The next() function returns the next item in an iterator.
        edge = next(x for x in ledges if x[0] == succ)
        print('succ:', succ)
        succ = edge[1]

    rev_path.append(succ)
    print('reverse path: ', rev_path)
    rev = rev_path[::-1]
    # print('path: ', rev)
    return rev


def plot_path(rgb, nrows, ncols, path, bfile):
    rgb1 = np.zeros((nrows, ncols, 3))
    # Paint in green the cells in the path
    for (r, c) in path:
        rgb1 = update_color(rgb, r, c, green)
    draw_board(nrows, ncols, rgb1, bfile)





if __name__ == '__main__':
    
    # Convert board cells to graph nodes
    # g = board2graph(3, 3)
    # Drawing a chess board (black and white)
    nrows = 5
    ncols = 5
    goal = (0, 0)
    print('Manhattan distance:', cityblock(goal, (3,3)))

    #     rgb = set_colors(nrows, ncols)
    #     draw_board(nrows, ncols, rgb)
    # Change one cell of the board to other color
    # rgb1 = update_color(rgb, 0, 1, green)
    # draw_board(nrows, ncols, rgb1)

    # # Convert from 2-Dimension indexes to 1-Dim and back
    #     k = convert2_1D(2, 3)
    #     print('1Dim index:', k)
    #     (i, j) = convert1_2D(k)
    #     print('2Dim index:', i, j)
    pass

