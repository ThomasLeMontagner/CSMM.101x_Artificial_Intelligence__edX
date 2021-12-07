import sys

import bts
from ac3 import AC3
from csp import CSP
from sudoku import Sudoku


def solver(given_sudoku_board):
    #print_sudoku(input_string)
    csp = CSP(given_sudoku_board)
    s = Sudoku(given_sudoku_board)

    result = ""
    # try AC3
    assignment = AC3(s)
    assignment.solve()
    if (assignment.is_solved()):
         result = assignment.get_result() + " AC3"

    # else BTS
    else:
        assignment = bts.backtracking_search(csp)
        result = bts.get_result(assignment) + " BTS"

    # write file
    f = open("output.txt", "w")
    f.write(result + "\n")
    f.close()


def main():
    input_string = sys.argv[1]
    solver(input_string)


if __name__ == '__main__':
    main()