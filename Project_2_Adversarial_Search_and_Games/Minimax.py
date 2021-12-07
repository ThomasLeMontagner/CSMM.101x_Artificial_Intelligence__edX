import time

from Helpers import *
import math

max_depth = 8 # maximum authorized depth
max_time = 0.05 # maximum computation time

def minimize(grid, current_depth, alpha, beta, start):
    #print(" -- minimize --")
    if is_terminal_state(grid) or current_depth >= max_depth or (time.clock() - start) >= max_time:
        #print("Depth: ", current_depth)
        return Eval(grid)

    minUtility = math.inf

    for child in get_next_computer_grids(grid):
        utility = maximize(child, current_depth +1, alpha, beta, start)

        if utility < minUtility:
            minUtility = utility

        if minUtility <= alpha:
            break

        if minUtility < beta:
            beta = minUtility

    #print("Minimize utility: ", minUtility)
    return minUtility


def maximize(grid, current_depth, alpha, beta, start):
    #print(" -- maximize --")
    if is_terminal_state(grid) or (time.clock() - start) >= max_time:# or current_depth >= max_depth:
        #print("Depth: ", current_depth)
        return Eval(grid)

    maxUtility = -math.inf

    for move in grid.getAvailableMoves():
        child_grid = grid.clone()
        child_grid.move(move)
        #print("MOVE: ", move)
        utility = minimize(child_grid, current_depth +1, alpha, beta, start)

        if utility > maxUtility:
            maxUtility = utility

        if maxUtility >= beta:
            break

        if maxUtility > alpha:
            alpha = maxUtility

    #print("Maximize utility: ", maxUtility)
    return maxUtility


def decision(grid):
    start = time.clock()
    return minimize(grid, 0, -math.inf, math.inf, start)