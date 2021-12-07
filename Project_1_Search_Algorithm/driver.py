"""
Skeleton code for Project 1 of Columbia University's AI EdX course (8-puzzle).
Python 3
"""
import queue as Q

import timeit
from collections import deque

import time

import resource

import sys

import math


#### SKELETON CODE ####

## The Class that Represents the Puzzle

class PuzzleState(object):
    """docstring for PuzzleState"""

    def __init__(self, config, n, parent=None, action="Initial", cost=0):

        if n * n != len(config) or n < 2:
            raise Exception("the length of config is not correct!")

        self.n = n

        self.cost = cost

        self.parent = parent

        self.action = action

        self.dimension = n

        self.config = config

        self.children = []

        for i, item in enumerate(self.config):

            if item == 0:
                self.blank_row = i // self.n

                self.blank_col = i % self.n

                break

    def display(self):

        for i in range(self.n):

            line = []

            offset = i * self.n

            for j in range(self.n):
                line.append(self.config[offset + j])

            print(line)

    def move_left(self):

        if self.blank_col == 0:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index - 1

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Left", cost=self.cost + 1)

    def move_right(self):

        if self.blank_col == self.n - 1:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index + 1

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Right", cost=self.cost + 1)

    def move_up(self):

        if self.blank_row == 0:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index - self.n

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Up", cost=self.cost + 1)

    def move_down(self):

        if self.blank_row == self.n - 1:

            return None

        else:

            blank_index = self.blank_row * self.n + self.blank_col

            target = blank_index + self.n

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Down", cost=self.cost + 1)

    def expand(self):

        """expand the node"""

        # add child nodes in order of UDLR

        if len(self.children) == 0:

            up_child = self.move_up()

            if up_child is not None:
                self.children.append(up_child)

            down_child = self.move_down()

            if down_child is not None:
                self.children.append(down_child)

            left_child = self.move_left()

            if left_child is not None:
                self.children.append(left_child)

            right_child = self.move_right()

            if right_child is not None:
                self.children.append(right_child)

        return self.children

# Function that Writes to output.txt

### Students need to change the method to have the corresponding parameters

def get_path_to_goal(puzzle_status):
    """ return path from start to goal """

    path = []
    while not puzzle_status.parent is None:
        path.insert(0,puzzle_status.action)
        puzzle_status = puzzle_status.parent
    return path

# return true if the config is in the list

def writeOutput(finalState, nodes_expanded, max_search_depth, runtime):
    """write output"""
    path_to_goal = get_path_to_goal(finalState)

    f= open("output.txt","w+")
    f.write("path_to_goal: "+  str(path_to_goal) + "\n")
    f.write("cost_of_path: " + str(len(path_to_goal)) + "\n")
    f.write("nodes_expanded: " + str(nodes_expanded) + "\n")
    f.write("search_depth: " + str(len(path_to_goal)) + "\n")
    f.write("max_search_depth: " + str(max_search_depth) + "\n")
    f.write("running_time: " + str(runtime) + "\n")
    f.write("max_ram_usage: " + "\n")

    f.close()

def bfs_search(initial_state):
    """BFS search"""
    start = timeit.default_timer()

    frontier = deque([initial_state])
    max_search_depth = 0

    explored = set()
    set_frontier = set()
    set_frontier.add(initial_state.config)

    while not len(frontier) == 0:
        # 1. remove from the queue
        state = frontier.popleft()

        # 2. Test against goal
        if test_goal(state):
            stop = timeit.default_timer()
            runtime = stop - start
            return writeOutput(state, len(explored), max_search_depth, runtime)

        explored.add(state.config)

        # 3. Expand
        for neighbor in state.expand():
            if (neighbor.config not in explored) and (neighbor.config not in set_frontier):
                frontier.append(neighbor)
                set_frontier.add(neighbor.config)

        if max_search_depth < state.cost:
            max_search_depth = state.cost + 1

        #print(len(explored), " -- ", "Expand: ", state.config)
        #print(len(frontier))


def dfs_search(initial_state):
    """DFS search"""
    start = timeit.default_timer()

    frontier = deque([initial_state])
    max_search_depth = 0

    explored = set()
    set_frontier = set()
    set_frontier.add(initial_state.config)

    while not len(frontier) == 0:
        # 1. remove from the queue
        state = frontier.pop()
        set_frontier.remove(state.config)

        # 2. Test against goal
        if test_goal(state):
            stop = timeit.default_timer()
            runtime = stop - start
            return writeOutput(state, len(explored), max_search_depth, runtime)

        explored.add(state.config)

        # 3. Expand
        neighboors = state.expand()
        neighboors.reverse()
        for neighbor in neighboors: # might need to revert the order
            if (neighbor.config not in explored) and (neighbor.config not in set_frontier):
                frontier.append(neighbor)
                set_frontier.add(neighbor.config)

        # 4. Depth
        if max_search_depth < state.cost:
            max_search_depth = state.cost +1

        #print(len(explored), " -- ", "Expand: ", state.config)
        #print(len(frontier))


def A_star_search(initial_state):
    """A * search"""
    start = timeit.default_timer()

    initial_state.cost = calculate_total_cost(initial_state)
    frontier = dict()
    frontier.setdefault(calculate_total_cost(initial_state), [initial_state])
    max_search_depth = 0

    explored = set()
    set_frontier = set()
    set_frontier.add(initial_state.config)

    while not len(frontier) == 0:
        # 1. remove from the queue
        state = get_min_cost_state(frontier)
        set_frontier.remove(state.config)

        # 2. Test against goal
        if test_goal(state):
            stop = timeit.default_timer()
            runtime = stop - start
            return writeOutput(state, len(explored), max_search_depth, runtime)

        explored.add(state.config)

        # 3. Expand
        for neighbor in state.expand(): # might need to revert the order
            if (neighbor.config not in explored) and (neighbor.config not in set_frontier):
                add_in_cost_order(neighbor, frontier)
                set_frontier.add(neighbor.config)

        # 4. Depth
        depth = len(get_path_to_goal(state))
        if max_search_depth < depth+1:
            max_search_depth = depth+1

        #print(len(explored), " -- ", "Expand: ", state.config)
        #print("Frontier: ", len(frontier))

def get_min_cost_state(dict):
    c = 9999999
    for cost, states in dict.items():
        if (not len(states) == 0) and cost < c:
            c = cost
    return dict[c].pop(0)

def add_in_cost_order(state, frontier):
    cost = state.cost + calculate_total_cost(state)
    try:
        frontier[cost].insert(len(frontier[cost]), state)
    except KeyError:
        frontier[cost] = [state]


def calculate_total_cost(state):
    """calculate the total estimated cost of a state"""
    n = int(math.sqrt(len(state.config)))
    
    h_cost = 0
    for i, item in enumerate(state.config):
        if item != 0:
            h_cost += calculate_manhattan_dist(i, item, n)

    g_cost = 0
    if state.parent is not None:
        g_cost = 1

    return g_cost + h_cost


def calculate_manhattan_dist(idx, value, n):
    """calculate the manhattan distance of a tile"""
    idx_row,idx_col = int(idx/ n) , idx % n
    goal_row,goal_col = int(value /n),value % n
    return abs(idx_row-goal_row) + abs(idx_col - goal_col)

def test_goal(puzzle_state):
    """test the state is the goal state or not"""

    for i, item in enumerate(puzzle_state.config):
        if i != item:
            return False
    return True


# Main Function that reads in Input and Runs corresponding Algorithm

def main():
    sm = sys.argv[1].lower()

    begin_state = sys.argv[2].split(",")

    begin_state = tuple(map(int, begin_state))

    size = int(math.sqrt(len(begin_state)))

    hard_state = PuzzleState(begin_state, size)

    if sm == "bfs":

        bfs_search(hard_state)

    elif sm == "dfs":

        dfs_search(hard_state)

    elif sm == "ast":

        A_star_search(hard_state)

    else:

        print("Enter valid command arguments !")


if __name__ == '__main__':
    main()
