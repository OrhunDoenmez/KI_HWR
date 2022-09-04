"""
queens.py
From Classic Computer Science Problems in Python Chapter 3
Copyright 2018 David Kopec

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from csp import Constraint, CSP
from typing import Dict, List, Optional

"""
A chessboard is an eight-by-eight grid of squares. 
A queen is a chess piece that can move on the chessboard 
any number of squares along any row, column, or diagonal. 
A queen is attacking another piece if, in a single move, 
it can move to the square the piece is on without jumping 
over any other piece. 
If the other piece is in the line of sight of the queen, 
then it’s attacked by it). 
The eight queens’ problem poses the question of how eight 
queens can be placed on a chessboard without any queen 
attacking another queen. 
"""

class QueensConstraint(Constraint[int, int]):
    def __init__(self, columns: List[int]) -> None:
        super().__init__(columns)
        self.columns: List[int] = columns

    """
    To solve the problem, we need a constraint that checks whether 
    any two queens are on the same row or diagonal (they were all 
    assigned different sequential columns, to begin with). 
    Checking for the same row is trivial but checking for the same 
    diagonal requires a little bit of math. 
    If any two queens are on the same diagonal, the difference 
    between their rows is the same as the difference between 
    their columns. 
    """
    def satisfied(self, assignment: Dict[int, int]) -> bool:
        # q1c = queen 1 column, q1r = queen 1 row
        for q1c, q1r in assignment.items():
            # q2c = queen 2 column
            for q2c in range(q1c + 1, len(self.columns) + 1):
                if q2c in assignment:
                    q2r: int = assignment[q2c] # q2r = queen 2 row
                    if q1r == q2r: # same row?
                        return False
                    if abs(q1r - q2r) == abs(q1c - q2c): # same diagonal?
                        return False
        return True # no conflict


if __name__ == "__main__":
    """
    To represent squares on the chess board, we’ll assign 
    each an integer row and an integer column. 
    We can ensure each of the eight queens isn’t on the same 
    column by assigning them sequentially the columns 1 through 8. 
    The variables in our constraint-satisfaction problem are the 
    column of the queen in question. 
    The domains can be the possible rows (again 1 through 8). 
    """
    columns: List[int] = [1, 2, 3, 4, 5, 6, 7, 8]
    rows: Dict[int, List[int]] = {}
    for column in columns:
        rows[column] = [1, 2, 3, 4, 5, 6, 7, 8]
    csp: CSP[int, int] = CSP(columns, rows)
    csp.add_constraint(QueensConstraint(columns))
    solution: Optional[Dict[int, int]] = csp.backtracking_search()
    if solution is None:
        print("No solution found!")
    else:
        print(solution)