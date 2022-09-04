"""
map_coloring.py
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
from pprint import pprint

"""
MapColoringConstraint isn’t generic in terms of type hinting,
but it subclasses a parameterized version of the generic class
Constraint that indicates both variables and domains are of type str.
"""
class MapColoringConstraint(Constraint[str, str]):
    def __init__(self, place1: str, place2: str) -> None:
        super().__init__([place1, place2])
        self.place1: str = place1
        self.place2: str = place2

    def satisfied(self, assignment: Dict[str, str]) -> bool:
        """
        If either place is not in the assignment then it is not
        yet possible for their colors to be conflicting
        :param assignment:
        :return:
        """
        if self.place1 not in assignment or self.place2 not in assignment:
            return True
        """
        check the color assigned to place1 is not the same as the
        color assigned to place2
        """
        return assignment[self.place1] != assignment[self.place2]


if __name__ == "__main__":
    """
    Imagine you have a map of Australia that you want to color by “regions”.
    No two adjacent regions should share a color.
    Can you color the regions with only three different colors?
    To model the problem as a CSP, we need to define the variables, domains,
    and constraints.
    The variables are the seven regions of Australia:
    Western Australia; Northern Territory; South Australia; Queensland;
    New South Wales; Victoria; and Tasmania.
    In our CSP, they can be modeled with strings.
    """
    variables: List[str] = ["Western Australia", "Northern Territory", "South Australia",
                            "Queensland", "New South Wales", "Victoria", "Tasmania"]
    """
    The domain of each variable is the three different colors that can
    possibly be assigned (we’ll use red, green, and blue).
    """
    domains: Dict[str, List[str]] = {}
    for variable in variables:
        domains[variable] = ["red", "green", "blue"]
    csp: CSP[str, str] = CSP(variables, domains)
    """
    The constraints are the tricky part.
    No two adjacent regions can be colored with the same color, and
    our constraints are dependent on which regions border one another.
    We can use binary constraints (constraints between two variables).
    Every two regions that share a border also share a binary constraint
    indicating they can’t be assigned the same color.
    
    The MapColoringConstraint is a subclass of the Constraint class
    to implement binary constraints in code. It takes two variables
    in its constructor: the two regions that share a border.
    Its overridden satisfied method checks whether the two regions
    both have a domain value (color) assigned to them —
    if either doesn’t, the constraint are trivially satisfied
    until they do (there isn't a conflict when one doesn’t yet have a color).
    Then it checks whether the two regions are assigned the same color
    (obviously there’s a conflict, meaning the constraint isn’t
    satisfied when they’re the same).
    """
    csp.add_constraint(MapColoringConstraint("Western Australia", "Northern Territory"))
    csp.add_constraint(MapColoringConstraint("Western Australia", "South Australia"))
    csp.add_constraint(MapColoringConstraint("South Australia", "Northern Territory"))
    csp.add_constraint(MapColoringConstraint("Queensland", "Northern Territory"))
    csp.add_constraint(MapColoringConstraint("Queensland", "South Australia"))
    csp.add_constraint(MapColoringConstraint("Queensland", "New South Wales"))
    csp.add_constraint(MapColoringConstraint("New South Wales", "South Australia"))
    csp.add_constraint(MapColoringConstraint("Victoria", "South Australia"))
    csp.add_constraint(MapColoringConstraint("Victoria", "New South Wales"))
    csp.add_constraint(MapColoringConstraint("Victoria", "Tasmania"))

    # Finally, backtracking_search() is called to find a solution.
    solution: Optional[Dict[str, str]] = csp.backtracking_search()
    if solution is None:
        print("No solution found!")
    else:
        pprint(solution)

"""
A correct solution includes an assigned color for every region.

{'Western Australia': 'red',
 'Northern Territory': 'green',
 'South Australia': 'blue',
 'Queensland': 'red',
 'New South Wales': 'green',
 'Victoria': 'red',
 'Tasmania': 'green'}
"""
