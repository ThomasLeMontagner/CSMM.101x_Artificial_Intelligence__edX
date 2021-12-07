from collections import deque
from copy import deepcopy


class AC3:
    def __init__(self, sudoku):
        self.sudoku = sudoku
        self.variables = self.initialize_variables(sudoku)
        self.possible_arcs = sudoku.get_arcs()

    def is_solved(self):
        for v in self.variables.values():
            if len(v) != 1:
                return False
        return True

    def solve(self):
        # print(self.arcs)
        queue = deque(deepcopy(self.possible_arcs))
        while not len(queue) == 0:
            arc = queue.popleft()
            xi = arc[0]
            xj = arc[1]
            if self.revise(xi, xj):  # remove comparison to 1
                if len(self.variables[xi]) == 0:
                    return False
                for xk in self.get_neighbors(xi, xj):
                    self.add_in_queue(xk, xi, queue)
        return True

    # Return True iff we revise the domain of xi
    def revise(self, xi, xj):
        revised = False
        for x in self.variables[xi]:
            if not self.contains_satisfying_value(x, xj):
                self.variables[xi].remove(x)
                revised = True
        return revised

    # Return true if xj contains at least one value satisfying the constraint regarding x
    def contains_satisfying_value(self, x, xj):
        for y in self.variables[xj]:
            if x != y:
                return True
        return False

    # Initialize board with all possible values
    def initialize_variables(self, sudoku):
        variables = deepcopy(sudoku.board)
        for k, v in variables.items():
            if v == 0:
                variables[k] = [i for i in range(1, 10)]
            else:
                variables[k] = [v]
        return variables

    # Return list of neighbors of a cell xi, except xj
    def get_neighbors(self, xi, xj):
        result = []
        for arc in self.possible_arcs:
            if arc[1] == xi and arc[0] != xj:
                result.append(arc[0])
            if arc[0] == xi and arc[1] != xj:
                result.append(arc[1])
        return result

    # Add the arc in the queue
    def add_in_queue(self, xk, xi, queue):
        if not self.is_arc_in_queue(xk, xi, queue):
            queue.append([xk, xi])

    # Return true if the arc in already in the queue
    def is_arc_in_queue(self, xk, xi, queue):
        for arc in queue:
            if arc[0] == xk and arc[1] == xi:
                return True
        return False

    # Print the result
    def get_result(self):
        result = ""
        for v in self.variables.values():
            result += str(v[0])
        return result
