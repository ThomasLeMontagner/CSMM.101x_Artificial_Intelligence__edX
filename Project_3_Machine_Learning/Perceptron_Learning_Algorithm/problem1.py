import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

# Run a PLA of a *.csv file and write output in *.csv file
def PLA(input_file = "input1.csv", output_file = "output1.csv"):
    # read file
    data = genfromtxt(input_file, delimiter=',')
    n = len(data[0])
    data = np.c_[np.ones(len(data)), data]
    data_pos = data[data[:,n] > 0][:,:n]
    data_neg = data[data[:,n] < 0][:,:n]

    #print(data_pos)
    #print(data_neg)

    # Initialization of weight at 0
    w = [0, 0, 0]
    weights = [[0, 0, 0]]

    while not has_converged(w, data): # test convergence
        for i in range(len(data)):
            y = data[i][n]
            x = list(data[i,:n])
            if y*calc_f(x, w) <= 0:
                update_weights(x,w,y)
                add_weight(weights, w)

    # show
    plotFile(data_pos, data_neg, weights)

    # Write csv file
    np.around(weights,3)
    np.savetxt(output_file, weights, fmt='%.2f',delimiter=",")

# return true is there is convergence
def has_converged(w,data):
    n = len(w)
    for i in range(len(data)):
        y = data[i][n]
        x = list(data[i,:n])
        if y*calc_f(x, w) <= 0:
            return False
    return True

# Calculate sum(w*x)
def calc_f(x,w):
    result = np.dot(w,x)
    if result <= 0:
        return -1
    else:
        return 1

# Update weights
def update_weights(x,w,y):
    for j in range(len(w)):
        w[j] = w[j] + x[j] * y
    return w

# Add weight in hte weights list for output w1, w2, b
def add_weight(weights, w):
    new_w = []
    for i in range(1, len(w)):
        new_w.append(w[i])
    new_w.append(w[0])
    weights.append(new_w)

# plot the data
def plotFile(data1, data2, weights):
    plt.scatter(data1[:,1], data1[:,2], c="blue", label='+1')
    plt.scatter(data2[:,1], data2[:,2], c="red", label='-1')

    x1 = np.linspace(0,15,100)
    for w in weights:
        if w[2] != 0:
            x2 = -(w[0] + w[1]*x1)/w[2]
            l = plt.plot(x1, x2, '-', c = "black", label='y=2x+1')
            del(plt.gca().lines[-1])

    plt.legend()
    plt.show()

# Main Function that reads in Input and Runs corresponding PLA

def main():
    input_file = sys.argv[1]

    output_file = sys.argv[2]

    PLA(input_file, output_file)


if __name__ == '__main__':
    main()