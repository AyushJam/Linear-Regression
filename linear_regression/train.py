"""
iBhubs CCBP: Linear Regression Project
Author - Ayush Jamdar
Date - 14.06.2022
"""

import numpy as np
import csv


def import_data():
    X = np.genfromtxt("train_X_lr.csv", delimiter=",", dtype=np.float64, skip_header=1)
    Y = np.genfromtxt("train_Y_lr.csv", delimiter=",", dtype=np.float64)
    return X, Y


def compute_cost(X, Y, W):
    xtheta = np.matmul(X, W)
    cost = np.matmul((xtheta - Y).T, xtheta - Y)

    return cost * (1 / (2 * X.shape[0]))


def compute_gradient_of_cost_function(X, Y, W):
    # gradient  = (1/m) XT . (X.theta - y)
    grad = (1 / X.shape[0]) * np.dot((X.T), (np.dot(X, W) - Y))
    return grad


def optimize_weights_using_gradient_descent(X, Y, W, threshold, learning_rate):
    prev_cost = 0
    iteration_num = 0
    while True:
        iteration_num += 1
        W = W - learning_rate * compute_gradient_of_cost_function(X, Y, W)
        cost = compute_cost(X, Y, W)[0]

        if abs(prev_cost - cost) < threshold:
            print(iteration_num, cost)
            break

        if iteration_num % 1e6 == 0:
            print(iteration_num, cost)

        prev_cost = cost
    return W


def train_model(X, Y):
    X = np.insert(X, 0, 1, axis=1)
    Y = Y.reshape(len(X), 1)
    W = np.zeros((X.shape[1], 1))
    threshold = 1e-6
    W = optimize_weights_using_gradient_descent(X, Y, W, threshold, 0.0002)

    return W


def save_model(weights, file_name):
    with open(file_name, "w") as weights_file:
        wr = csv.writer(weights_file)
        wr.writerows(weights)
        weights_file.close()


if __name__ == "__main__":
    X, Y = import_data()
    weights = train_model(X, Y)
    save_model(weights, "WEIGHTS_FILE.csv")
