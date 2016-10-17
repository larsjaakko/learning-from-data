#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

SAMPLESIZE = 100
RUNS = 1000


def create_target():

    targetA = np.random.uniform(-1, 1, 2)
    targetB = np.random.uniform(-1, 1, 2)

    target = [targetB[0] - targetA[0], targetB[1] - targetA[1]]

    return targetA, targetB, target


def create_point_vectors(points, targetA, targetB, target):

    vectors = np.zeros([SAMPLESIZE, 2, 2])

    vectors[:, 0, 0] = points[:, 0] - targetA[0]
    vectors[:, 0, 1] = points[:, 1] - targetA[1]
    vectors[:, 1] = target

    return vectors


def evaluate_sign(vectors):

    Y = np.zeros(SAMPLESIZE)

    Y = np.sign(np.linalg.det(vectors))

    return Y


def create_w():

    w = np.zeros((3, 1))

    return w


def select_random_point(points, Y, H):
    # This expects the points vector to still contain some misclassified points
    # A function to check whether classification is complete is required.

    found = False

    while not found:

        sampleIndex = np.random.choice(SAMPLESIZE, 1)

        if Y[sampleIndex] != H[sampleIndex]:

            found = True

            sample = points[sampleIndex]

    return sample, sampleIndex


def count_misclassified(H, Y):
    # Will return the number of points that are still misclassified

    count = 0

    for index, x in np.ndenumerate(Y):

        if (Y[index] != H[index]):

                count += 1

    return count


def create_H(X, w):

    wt = np.transpose(w)
    H = np.dot(wt, X)
    H = np.sign(H[0])

    return H


def update_weights(w, sample, sampleIndex, Y):

    adjustment = Y[sampleIndex] * sample

    w += np.transpose(adjustment)

    return w


def create_points(number):

    points = np.random.uniform(-1, 1, (number, 2))

    # Add column of x0 = 1

    points = np.c_[np.ones(number), points]

    # NB! The points matrix has to be transposed,
    # it is not in the correct orientation for calculations

    return points


def create_X(points):

    X = np.transpose(points)

    return X


def learn(run):

    # Create random dataset of SAMPLESIZE samples
    points = create_points(SAMPLESIZE)

    # Create target function for the learning algorithm to estimate
    targetA, targetB, target = create_target()

    # Create vectors to use in classification of points
    vectors = create_point_vectors(points, targetA, targetB, target)

    # Classify all points according to the target function,
    # place them in a row vector
    Y = evaluate_sign(vectors)

    # Set up X and w and create the initial H
    X = create_X(points)
    w = create_w()
    H = create_H(X, w)

    iterations = 0

    # Check if H is correctly classified
    # If more than 0 points are misclassified, the loop continues

    while count_misclassified(H, Y) != 0:

        # Select random point for updating
        sample, sampleIndex = select_random_point(points, Y, H)
        w = update_weights(w, sample, sampleIndex, Y)

        # Create new H vector

        H = create_H(X, w)

        iterations += 1

    print("Run # ",
          run,
          " had ",
          iterations,
          " iterations to classify all points.")

    return iterations


def learn_and_measure():

    return


def main():

    # Create empty array to hold number of iterations required
    # per RUNS

    iterations = np.zeros(RUNS)

    for i in range(RUNS):

        iterations[i] = learn(i)

    print()
    print("The average number of iterations was ", np.mean(iterations))

if __name__ == "__main__":
    main()
