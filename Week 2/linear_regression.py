#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

SAMPLESIZE = 100


def create_target():

    targetA = np.random.uniform(-1, 1, 2)
    targetB = np.random.uniform(-1, 1, 2)

    target = [targetB[0] - targetA[0], targetB[1] - targetA[1]]

    return targetA, targetB, target


def create_points(number):

    points = np.random.uniform(-1, 1, (number, 2))

    # Add column of x0 = 1

    points = np.c_[np.ones(number), points]

    # NB! The points matrix has to be transposed,
    # it is not in the correct orientation for calculations

    return points


def create_point_vectors(points, targetA, targetB, target, size):

    vectors = np.zeros([size, 2, 2])

    vectors[:, 0, 0] = points[:, 0] - targetA[0]
    vectors[:, 0, 1] = points[:, 1] - targetA[1]
    vectors[:, 1] = target

    return vectors


def evaluate_sign(vectors, size):

    Y = np.zeros(size)

    Y = np.sign(np.linalg.det(vectors))

    Y = np.transpose(Y)

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

    # Create target function for the learning algorithm to estimate
    targetA, targetB, target = create_target()

    # Create vectors to use in classification of points
    vectors = create_point_vectors(points, targetA, targetB, target, SAMPLESIZE)

    # Classify all points according to the target function,
    # place them in a row vector
    Y = evaluate_sign(vectors, SAMPLESIZE)

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

    return np.sum(results), w, targetA, targetB, target


def validate(weights, freshpoints, targetA, targetB, target):

    vectors = create_point_vectors(freshpoints, targetA, targetB, target, 1000)
    Y = evaluate_sign(vectors, 1000)
    X = create_X(freshpoints)

    outresults = np.zeros(1000)

    for index in range(1000):

        H = create_H(X, weights[index])

        results = evaluate_regression(Y, H)

        outresults[index] = np.mean(results)

    return outresults


def main():

    results = np.zeros(1000)
    weights = np.zeros(shape=(1000, 3, 1))

    for index in range(1000):

        print('Iteration no. ', index)

        results[index], weights[index], targetA, targetB, target = learn()

    print("E-in equals: ", np.mean(results)/100)

    freshpoints = create_points(1000)

    outresults = validate(weights, freshpoints, targetA, targetB, target)

    print("E-out equals: ", np.mean(outresults)/100)

if __name__ == "__main__":
    main()
