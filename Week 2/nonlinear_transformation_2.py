#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

SAMPLESIZE = 1000


def create_points(number):

    points = np.random.uniform(-1, 1, (number, 2))

    points = create_noise(points)

    # Add column of x0 = 1

    points = np.c_[np.ones(number), points]

    # Add additional columns as per exercise text
    points = np.c_[points,
                   points[:, 1]*points[:, 2],
                   points[:, 1] ** 2,
                   points[:, 2] ** 2
                   ]

    # NB! The points matrix has to be transposed,
    # it is not in the correct orientation for calculations
    # Oh wait it is..

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

    # Y = np.sign(-1 - 0.05 * points[:, 1] + 0.08 * points[:, 2] + 0.13 * points[:, 3] + 1.5 * points[:, 4] + 1.5 * points[:, 5])
    # Y = np.sign(-1 - 0.05 * points[:, 1] + 0.08 * points[:, 2] + 0.13 * points[:, 3] + 1.5 * points[:, 4] + 15 * points[:, 5])
    # Y = np.sign(-1 - 0.05 * points[:, 1] + 0.08 * points[:, 2] + 0.13 * points[:, 3] + 15 * points[:, 4] + 1.5 * points[:, 5])
    # Y = np.sign(-1 - 1.5 * points[:, 1] + 0.08 * points[:, 2] + 0.13 * points[:, 3] + 0.05 * points[:, 4] + 0.05 * points[:, 5])
    # Y = np.sign(-1 - 0.05 * points[:, 1] + 0.08 * points[:, 2] + 1.5 * points[:, 3] + 15 * points[:, 4] + 15 * points[:, 5])

    # print('Y er : ', Y)

    return Y


def create_X(points):

    # X = np.transpose(points)
    X = points

    return X


def create_w():

    w = np.zeros(shape=(6, 1))

    return w


def create_pseudo_inverse(X):

    XT = np.transpose(X)
    pseudo = np.dot(np.linalg.inv(np.dot(XT, X)), XT)

    return pseudo


def update_weights(w, pseudo, Y):

    w = np.dot(pseudo, Y).reshape(6, 1)

    return w


def create_H(X, w):

    wt = np.transpose(w)

    # Remember to transpose X back, since the x vectors are column vectors
    H = np.dot(wt, X.T)
    H = np.sign(H[0])

    return H


def evaluate_regression(Y, H):

    results = np.zeros(1000)

    for index in range(1000):

        if Y[index] != H[index]:

            results[index] = 1

    return results


def validate(weights, freshpoints):

    Y = evaluate_sign(freshpoints)
    X = create_X(freshpoints)

    H = create_H(X, weights)

    results = evaluate_regression(Y, H)

    return np.mean(results)


def compare_hypotheses(points, weights):

    size_total = np.shape(points)
    size = max(size_total)

    # g = np.zeros(size)

    # Y = np.sign(points[:, 1] ** 2 + points[:, 2] ** 2 - 0.6)
    X = create_X(points)
    H = create_H(X, weights)

    # g = np.sign(-1 - 0.05 * points[:, 1] + 0.08 * points[:, 2] + 0.13 * points[:, 3] + 1.5 * points[:, 4] + 1.5 * points[:, 5])
    # g = np.sign(-1 - 0.05 * points[:, 1] + 0.08 * points[:, 2] + 0.13 * points[:, 3] + 1.5 * points[:, 4] + 15 * points[:, 5])
    # g = np.sign(-1 - 0.05 * points[:, 1] + 0.08 * points[:, 2] + 0.13 * points[:, 3] + 15 * points[:, 4] + 1.5 * points[:, 5])
    # g = np.sign(-1 - 1.5 * points[:, 1] + 0.08 * points[:, 2] + 0.13 * points[:, 3] + 0.05 * points[:, 4] + 0.05 * points[:, 5])
    g = np.sign(-1 - 0.05 * points[:, 1] + 0.08 * points[:, 2] + 1.5 * points[:, 3] + 15 * points[:, 4] + 15 * points[:, 5])

    correct = 0

    for index in range(size):

        if g[index] == H[index]:

            correct += 1


    correct_fraction = (float(correct) / size)

    return correct_fraction



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
    weights = np.zeros(shape=(1000, 6, 1))
    outresults = np.zeros(1000)
    comparisons = np.zeros(1000)

    for index in range(1000):

        print('Iteration no. ', index+1)

        results[index], weights[index] = learn()

        freshpoints = create_points(1000)

        comparisons[index] = compare_hypotheses(freshpoints, weights[index])

        outresults[index] = validate(weights[index], freshpoints)

    print("Comparison with g gives: ", np.mean(comparisons))
    print("E-out is: ", np.mean(outresults))


if __name__ == "__main__":
    main()
