"""
csp.py
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
from typing import Generic, TypeVar, Dict, List, Optional
from abc import ABC, abstractmethod

V = TypeVar('V') # variable type
D = TypeVar('D') # domain type

"""
Building a constraint-satisfaction problem framework

Constraints are defined using a Constraint class.
Each Constraint consists of the variables it constrains and
a method that checks whether it’s satisfied.
The default implementation must be overridden because
the Constraint class is defined as an abstract base class.
Abstract base classes aren’t meant to be instantiated;
only the subclasses that override and implement their
@abstractmethods are used.
Abstract base classes serve as templates for a class hierarchy.
many of the collection classes in Python’s standard library
are implemented via abstract base classes. The general advice
is not to use them in your own code unless you’re sure that
you’re building a framework upon which others will build, and
not only an internal use class hierarchy.

Base class for all constraints
"""

class Constraint(Generic[V, D], ABC):
    # The variables that the constraint is between
    def __init__(self, variables: List[V]) -> None:
        self.variables = variables

    # Must be overridden by subclasses
    @abstractmethod
    def satisfied(self, assignment: Dict[V, D]) -> bool:
        ...

"""
The centerpiece of this constraint-satisfaction framework is
a class called CSP.
CSP is the gathering point for variables, domains, and constraints.
In terms of its type hints, it uses generics to make itself
flexible enough to work with any kind of variables and domain
values (V keys and D domain values).
Within CSP, the definitions of the collection’s variables,
domains, and constraints are:
The variables collection is a list of variables,
domains is a dict mapping variables to lists of possible values
(the domains of those variables), and
constraints is a dict that maps each variable to a list of the
constraints imposed on it.
A constraint satisfaction problem consists of 
variables of type V that have ranges of values known as domains of type D 
and constraints that determine whether a particular variable's domain 
selection is valid
"""
class CSP(Generic[V, D]):
    # The __init__() initializer creates the constraints dict.
    def __init__(self, variables: List[V], domains: Dict[V, List[D]]) -> None:
        self.variables: List[V] = variables # variables to be constrained
        self.domains: Dict[V, List[D]] = domains # domain of each variable
        self.constraints: Dict[V, List[Constraint[V, D]]] = {}
        for variable in self.variables:
            self.constraints[variable] = []
            if variable not in self.domains:
                raise LookupError("Every variable should have a domain assigned to it.")

    """
    The add_constraint() method goes through all of the variables
    touched by a given constraint and adds itself to the constraints
    mapping for each of them.
    """
    def add_constraint(self, constraint: Constraint[V, D]) -> None:
        for variable in constraint.variables:
            if variable not in self.variables:
                raise LookupError("Variable in constraint not in CSP")
            else:
                self.constraints[variable].append(constraint)
    """
    How do we know if a given configuration of variables and
    selected domain values satisfy the constraints?
    We’ll call such a given configuration an “assignment.”
    We need a function that checks every constraint for a given
    variable against an assignment to see if the variable’s value
    in the assignment works for the constraints.
    Check if the value assignment is consistent by checking all constraints
    for the given variable against it
    """
    def consistent(self, variable: V, assignment: Dict[V, D]) -> bool:
        # consistent goes through every constraint for a given variable
        # (it’s always a variable that was newly added to the assignment)
        # and checks if the constraint is satisfied, given the new assignment.
        for constraint in self.constraints[variable]:
            if not constraint.satisfied(assignment):
                return False
        return True

    """
    This constraint-satisfaction framework uses a simple backtracking
    search to find solutions to problems.
    The following backtracking search function is a kind of recursive
    depth-first search.
    """
    def backtracking_search(self, assignment: Dict[V, D] = {}) -> Optional[Dict[V, D]]:
        # The base case for the recursive search is finding a valid assignment
        # for every variable. Once we have, we return the first instance of a
        # solution that was valid i.e., we stop searching
        # assignment is complete if every variable is assigned (our base case)
        if len(assignment) == len(self.variables):
            return assignment

        # To select a new variable whose domain we can explore, we go through
        # all of the variables and find the first that doesn’t have an assignment.
        # To do this, we create a list of variables in self.variables but
        # not in assignment through a list comprehension, and call it
        # unassigned.
        # get all variables in the CSP but not in the assignment
        unassigned: List[V] = [v for v in self.variables if v not in assignment]

        # Then we pull out the first value is unassigned.
        # get the every possible domain value of the first unassigned variable
        first: V = unassigned[0]
        # We try assigning every possible domain value for that variable,
        # one at a time. The new assignment for each is stored in a local
        # dictionary called local_assignment.
        for value in self.domains[first]:
            local_assignment = assignment.copy()
            local_assignment[first] = value
            # If the new assignment in local_assignment is consistent with all
            # of the constraints (which is what consistent() checks for),
            # we continue recursively searching with the new assignment in place.
            # If the new assignment turns out to be complete (the base case),
            # we return the new assignment up the recursion chain.
            # if we're still consistent, we recurse (continue)
            if self.consistent(first, local_assignment):
                # NOTE: you are claiming that result is meant to contain optionaly
                # a dictionary where the keys have type V and the values have type D
                # In any case, result will contain the output from self.backtracking_search
                result: Optional[Dict[V, D]] = self.backtracking_search(local_assignment)
                # if we didn't find the result, we will end up backtracking
                if result is not None:
                    return result
        # Finally, if we’ve gone through every possible domain value for
        # a particular variable, and there’s no solution utilizing the existing
        # set of assignments, we return None, indicating no solution.
        # This leads to backtracking up the recursion chain to the point
        # where a different prior assignment could have been made.
        return None
