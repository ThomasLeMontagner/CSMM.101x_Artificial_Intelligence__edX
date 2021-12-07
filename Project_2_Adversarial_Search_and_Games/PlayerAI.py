from BaseAI import BaseAI
import time
from Minimax import *

class PlayerAI(BaseAI):

    def getMove(self, grid):
            maxUtility = -math.inf
            nextMove = -1
            startTime = time.clock()

            for move in grid.getAvailableMoves():
                print("MOVE: ", move)
                child = grid.clone()
                child.move(move)

                utility = decision(child)
                print("utility: ", utility)
                if utility >= maxUtility:
                    maxUtility = utility
                    nextMove = move
            endTime = time.clock()
            print("Time: ", endTime - startTime)
            return nextMove

