#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

SAMPLESIZE = 1000


def create_points(number):

    points = np.random.uniform(-1, 1, (number, 2))

    points = create_noise(points)

    # Add column of x0 = 1

    points = np.c_[np.ones(number), points]

    # NB! The points matrix has to be transposed,
    # it is not in the correct orientation for calculations

    return points


def create_noise(points):

    iterator = 0
    size_total = np.shape(points)
    size = max(size_total)
    noise = size/10

    while iterator < (noise):

        sampleIndex = np.random.choice(size, 1)

        points[sampleIndex] = points[sampleIndex] * (-1)

        iterator += 1

    return points


def evaluate_sign(points):

    size_total = np.shape(points)
    size = max(size_total)

    Y = np.zeros(size)

    Y = np.sign(points[:, 1] ** 2 + points[:, 2] ** 2 - 0.6)

    # print('Y er : ', Y)

    return Y


def create_X(points):

    # X = np.transpose(points)
    X = points

    return X


def create_w():

    w = np.zeros(shape=(3, 1))

    return w


def create_pseudo_inverse(X):

    XT = np.transpose(X)
    pseudo = np.dot(np.linalg.inv(np.dot(XT, X)), XT)

    return pseudo


def update_weights(w, pseudo, Y):

    w = np.dot(pseudo, Y).reshape(3, 1)

    return w


def create_H(X, w):

    wt = np.transpose(w)

    # Remember to transpose X back, since the x vectors are column vectors
    H = np.dot(wt, X.T)
    H = np.sign(H[0])

    return H


def evaluate_regression(Y, H):

    results = np.zeros(100)

    for index in range(100):

        if Y[index] != H[index]:

            results[index] = 1

    return results


def learn():

    # Create random dataset of SAMPLESIZE samples
    points = create_points(SAMPLESIZE)

    # Classify all points according to the target function,
    # place them in a row vector
    Y = evaluate_sign(points)

    # Set up X and w
    # Although points should alredy be correct for X?
    # No, we need to multiply w with the x vectors, which are transposed
    # in large X

    X = create_X(points)
    w = create_w()

    # Update the weights with the pseudo-inverse of X
    pseudo = create_pseudo_inverse(X)
    w = update_weights(w, pseudo, Y)

    H = create_H(X, w)

    results = evaluate_regression(Y, H)

    return np.mean(results), w


def main():

    results = np.zeros(1000)
    weights = np.zeros(shape=(1000, 3, 1))

    for index in range(1000):

        print('Iteration no. ', index+1)

        results[index], weights[index] = learn()

    print("E-in equals: ", np.mean(results))

    # freshpoints = create_points(1000)

    # outresults = validate(weights, freshpoints, targetA, targetB, target)

    # print("E-out equals: ", np.mean(outresults)/100)

if __name__ == "__main__":
    main()
