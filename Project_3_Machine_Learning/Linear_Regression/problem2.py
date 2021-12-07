import math
import sys
import numpy as np

# Run a LR of a *.csv file and write output in *.csv file
from sklearn.preprocessing import scale

def add_output(outputs, alpha, iterations, beta):
    new_line = [alpha, iterations]
    for i in beta:
        new_line.append(i)
    outputs.append(new_line)


def LR(input_file = "input2.csv", output_file = "output2.csv"):
    # read file
    data = np.genfromtxt(input_file, delimiter=',')
    X = data[:,:2]
    y = data[:,2]

    # scale X and add column of 1
    X[:,0] = scale(X[:,0], 0, with_mean=True, with_std=True)
    X[:,1] = scale(X[:,1], 0, with_mean=True, with_std=True)
    X = np.c_[np.ones(len(X)), X]

    own_alpha = 0.02
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, own_alpha]

    iterations = 100

    outputs = []
    for alpha in alphas:
        beta = [0 for i in range(len(X[0]))]
        for it in range(iterations):
            z = np.dot(X, beta)
            h = z #sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / len(y)
            beta -= alpha * gradient
        #print("alpha: ", alpha)
        #print("beta: ",  beta)

        add_output(outputs, alpha, iterations, beta)

    np.savetxt(output_file, outputs,delimiter=",")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


def main():
    input_file = sys.argv[1]

    output_file = sys.argv[2]

    LR(input_file, output_file)


if __name__ == '__main__':
    main()