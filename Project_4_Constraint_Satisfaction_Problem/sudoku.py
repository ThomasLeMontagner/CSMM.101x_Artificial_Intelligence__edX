class Sudoku:
    def __init__(self, board_list):
        self.board = self.convert_list_to_dict(board_list)

    # Convert a list of 81 interger to dictionary
    def convert_list_to_dict(self, board_list):
        rows = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
        columns = [i for i in range(1,10)]
        board = dict()

        if check_list(board_list):
            i = 0
            for r in rows:
                for c in columns:
                    cell = r + str(c)
                    board[cell] = int(board_list[i])
                    i += 1
        return board

    # Return all possible arcs
    def get_arcs(self):
        arcs = []
        all = self.get_lines() + self.get_columns() + self.get_squares()

        for element in all:
            for i in range(0,8):
                for j in range(i+1, 9):
                    arc1 = [element[i], element[j]]
                    if not [element[i], element[j]] in arcs:
                        arcs.append(arc1)
                    arc2 = [element[j], element[i]]
                    if not [element[j], element[i]] in arcs:
                        arcs.append(arc2)
        return arcs

    # get all lines of the sudoku
    def get_lines(self):
        return [["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"],
                ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"],
                ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"],
                ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"],
                ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"],
                ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"],
                ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9"],
                ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9"],
                ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9"]]

    # get all columns of the sudoku
    def get_columns(self):
        return [["A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1", "I1"],
                ["A2", "B2", "C2", "D2", "E2", "F2", "G2", "H2", "I2"],
                ["A3", "B3", "C3", "D3", "E3", "F3", "G3", "H3", "I3"],
                ["A4", "B4", "C4", "D4", "E4", "F4", "G4", "H4", "I4"],
                ["A5", "B5", "C5", "D5", "E5", "F5", "G5", "H5", "I5"],
                ["A6", "B6", "C6", "D6", "E6", "F6", "G6", "H6", "I6"],
                ["A7", "B7", "C7", "D7", "E7", "F7", "G7", "H7", "I7"],
                ["A8", "B8", "C8", "D8", "E8", "F8", "G8", "H8", "I8"],
                ["A9", "B9", "C9", "D9", "E9", "F9", "G9", "H9", "I9"]]

    # get all squares of the sudoku
    def get_squares(self):
        return [["A1", "B1", "C1", "A2", "B2", "C2", "A3", "B3", "C3"],
                ["A4", "B4", "C4", "A5", "B5", "C5", "A6", "B6", "C6"],
                ["A7", "B7", "C7", "A8", "B8", "C8", "A9", "B9", "C9"],
                ["D1", "E1", "F1", "D2", "E2", "F2", "D3", "E3", "F3"],
                ["D4", "E4", "F4", "D5", "E5", "F5", "D6", "E6", "F6"],
                ["D7", "E7", "F7", "D8", "E8", "F8", "D9", "E9", "F9"],
                ["G1", "H1", "I1", "G2", "H2", "I2", "G3", "H3", "I3"],
                ["G4", "H4", "I4", "G5", "H5", "I5", "G6", "H6", "I6"],
                ["G7", "H7", "I7", "G8", "H8", "I8", "G9", "H9", "I9"]]


# return true if list length is 81
def check_list(board_list):
    if len(board_list) != 81:
        print("Length = ", len(board_list), ". The length of the list is not correct.")
        return False
    return True
