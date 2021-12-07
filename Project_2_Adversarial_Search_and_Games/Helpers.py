import math
import numpy as np


# calculate the utility of a Grid
def Eval(grid):
    num_list = []
    for line in grid.map:
        num_list = num_list + line

    # SCORES
    freeTiles = getFreeTiles(grid)
    smoothness = smoothnessScore(grid)
    mono = monotonicityScore(grid)
    weightedPos = positionScore2(grid)
    maxTile = grid.getMaxTile()

    # WEIGHTS
    w_freeTile = 10
    w_smoothness = 5 # 5 worked well with w_freeTile = 1 and w_mono = 1
    w_mono = 2
    w_weightedPos = 2

    # PRINT SCORES
    #print(" __ ")
    #print("FreeTiles: ", freeTiles*w_freeTile)
    #print("Smoothness: ", smoothness*w_smoothness)
    #print("Monotonicity: ", mono*w_mono)
    #print("WeightedPos: ", math.sqrt(weightedPos)*w_weightedPos)

    if grid.map[0][0] == maxTile:
        eval = freeTiles*w_freeTile + smoothness*w_smoothness + mono*w_mono + math.sqrt(weightedPos)*10*w_weightedPos
    else:
        eval = freeTiles*w_freeTile + smoothness*w_smoothness + mono*w_mono + math.sqrt(weightedPos)*w_weightedPos
    #print("eval: ", eval)

    return eval

def isMaxinCorner(grid):
    m = grid.getMaxTile()
    return (grid.map[0][0] == m or grid.map[0][3] == m or grid.map[3][0] == m or grid.map[3][3] == m)


# return number of free tiles
def getFreeTiles(grid):
    result = 0
    for line in grid.map:
        result += line.count(0)
    return result


# Return smoothness of the board
def smoothnessScore(grid):
    smoothness = 0

    for line in range(4):
        for column in range(4):
            if grid.map[line][column] != 0:
                # smoothnessScore1 on lines
                if column < 3 and grid.map[line][column] == grid.map[line][column + 1]:
                    smoothness += grid.map[line][column]
                # smoothnessScore1 on columns
                if line < 3 and grid.map[line][column] == grid.map[line+1][column]:
                    smoothness += grid.map[line][column]
    if smoothness == 0:
        smoothness = 1
    return math.sqrt(smoothness)

def monotonicityScore(grid):
    l1 = grid.map[0]
    l2 = grid.map[1]
    l2.reverse()
    l3 = grid.map[2]
    l4 = grid.map[3]
    l4.reverse()

    l = l1 + l2 + l3 + l4
    s1 = 0
    for i in range(15):
        curr = 0
        next = 0
        if l[i] != 0:
            #curr = math.log(l[i])/math.log(2)
            curr = math.sqrt(l[i])

        if l[i+1] != 0:
            #next = math.log(l[i+1])/math.log(2)
            next = math.sqrt(l[i+1])
        s1 += curr - next

    s2 = 0
    m = np.array(grid.map)
    c1 = list(m[:,0])
    c2 = list(m[:,1])
    c3 = list(m[:,2])
    c4 = list(m[:,3])
    c2.reverse()
    c4.reverse()
    c = c1 + c2 + c3 + c4
    for i in range(3):
        curr = 0
        next = 0
        if c[i] != 0:
            #curr = math.log(c[i])/math.log(2)
            curr = math.sqrt(c[i])
        if c[i+1] != 0:
            #next = math.log(c[i+1])/math.log(2)
            next = math.sqrt(c[i+1])
        s2 += curr - next

    return max(s1, s2)

def positionScore2(grid):
    emptyTiles = getFreeTiles(grid)
    maxTile = grid.getMaxTile()
    Ord1 = 0
    Ord2 = 0

    weights1 = [[65536,32768,16384,8192],[512,1024,2048,4096],[256,128,64,32],[2,4,8,16]]
    weights2 = [[65536,512,256, 2],[32768,1024, 128, 4],[16384,2048, 64,8],[8192,4096, 32, 16]]
    if maxTile == grid.map[0][0]:
        Ord1 += (math.log(grid.map[0][0])/math.log(2))*weights1[0][0]
        Ord2 += (math.log(grid.map[0][0])/math.log(2))*weights2[0][0]
    for i in range(4):
        for j in range(4):
            if grid.map[i][j] >= 8:
                Ord1 += weights1[i][j]*(math.log(grid.map[i][j])/math.log(2))
                Ord2 += weights2[i][j]*(math.log(grid.map[i][j])/math.log(2))
    if Ord1 ==0 and Ord2 ==0:
        return 1
    return max(Ord1, Ord2)/(16-emptyTiles)

def positionScore(grid):
    #d= [[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]]
    w1 = [[135,121,102,99],[75,76,88,79],[60,56,37,16],[12,9,5,3]]
    w2 = [[2048,1024,512,256],[128,64,32,16],[8,7,6,5],[4,3,2,1]]
    w3 = [[500,400,300,256],[128,64,32,16],[8,7,6,5],[4,3,2,1]]
    w4 = [[2048, 1024, 64, 32], [512, 128, 16, 2], [256, 8, 2, 1],[4, 2, 1, 1]]
    w5 = [[5,4,4,3],[3,3,3,3],[2,2,1,1],[2,1,1,1]]
    w6 = [[2,1,1,2],[1,0,0,1],[1,0,0,1],[2,1,1,2]]
    result = 0
    freeTiles = getFreeTiles(grid)
    for i in range(4):
        for j in range(4):
            #result += grid.map[i][j]*w[i][j]*freeTiles/d[i][j]
            result += grid.map[i][j]*w6[i][j] #*freeTiles
            #result += grid.map[i][j]*w[i][j]
    return result

# Return true is the grid is a terminal state:
# - no move available
def is_terminal_state(grid):
    return not grid.canMove()

# Return list of possible random grids
def get_next_computer_grids(grid):
    grids = []
    for tile in grid.getAvailableCells():
        grid2 = grid.clone()
        grid4 = grid.clone()
        grid2.insertTile(tile, 2)
        grid4.insertTile(tile, 4)
        grids.append(grid2)
        grids.append(grid4)
    return grids


