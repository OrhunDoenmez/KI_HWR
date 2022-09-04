"""
send_more_money.py
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
SEND+MORE=MONEY is a cryptarithmetic puzzle, meaning 
it’s about finding digits that replace letters to make 
a mathematical statement true. 
Each letter in the problem represents one digit (0–9). 
No two letters can represent the same digit. 
When a letter repeats, it means a digit repeats in the 
solution.
"""

class SendMoreMoneyConstraint(Constraint[str, int]):
    def __init__(self, letters: List[str]) -> None:
        super().__init__(letters)
        self.letters: List[str] = letters

    """
    SendMoreMoneyConstraint’s satisfied() method does a few things. 
    First, it checks if there are any letters representing the same digits. 
    If there are, it’s an invalid solution, and it returns False. 
    Next, it checks if all letters have been assigned. 
    If they have, it checks to see if the formula (SEND+MORE=MONEY) 
    is correct with the given assignment. 
    If it is, a solution has been found, and it returns True. 
    Otherwise, it returns False. 
    Finally, if all letters haven’t yet been assigned, it returns True. 
    This is to ensure that a partial solution continues to be worked on.
    """
    def satisfied(self, assignment: Dict[str, int]) -> bool:
        # if there are duplicate values then it's not a solution
        if len(set(assignment.values())) < len(assignment):
            return False

        # if all variables have been assigned, check if it adds correctly
        if len(assignment) == len(self.letters):
            s: int = assignment["S"]
            e: int = assignment["E"]
            n: int = assignment["N"]
            d: int = assignment["D"]
            m: int = assignment["M"]
            o: int = assignment["O"]
            r: int = assignment["R"]
            y: int = assignment["Y"]
            send: int = s * 1000 + e * 100 + n * 10 + d
            more: int = m * 1000 + o * 100 + r * 10 + e
            money: int = m * 10000 + o * 1000 + n * 100 + e * 10 + y
            return send + more == money
        return True # no conflict


if __name__ == "__main__":
    """
    Note that we preassigned the answer for the letter M. 
    This was to ensure that the answer doesn’t include a 
    zero for M because our constraint has no notion of the 
    concept that a number can’t start with zero. 
    Feel free to try it out without that preassigned answer.
    The solution should look something like this:  
    {'S': 9, 'E': 5, 'N': 6, 'D': 7, 'M': 1, 'O': 0, 'R': 8, 'Y': 2}
    """
    letters: List[str] = ["S", "E", "N", "D", "M", "O", "R", "Y"]
    possible_digits: Dict[str, List[int]] = {}
    for letter in letters:
        possible_digits[letter] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    possible_digits["M"] = [1]  # so we don't get answers starting with a 0
    csp: CSP[str, int] = CSP(letters, possible_digits)
    csp.add_constraint(SendMoreMoneyConstraint(letters))
    solution: Optional[Dict[str, int]] = csp.backtracking_search()
    if solution is None:
        print("No solution found!")
    else:
        print(solution)