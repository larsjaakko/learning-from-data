import numpy as np

SAMPLESIZE = 10
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

    print("Her er Y-matrisen: ", Y)

    return Y


def create_w():

    w = np.zeros(3)

    return w


def select_random_point(points, Y, H):
    # This expects the points vector to still contain some misclassified points
    # A function to check whether classification is complete is required.

    found = False

    while not found:

        sampleIndex = np.random.choice(10, 1)

        if Y[sampleIndex] is not H[sampleIndex]:

            found = True

            sample = points[sampleIndex]

    return sample, sampleIndex


def count_misclassified(H, Y):
    # Will return the number of points that are still misclassified

    count = 0

    for index, x in numpy.ndenumerate(Y):

        if Y[index] is not H[index]:

                count += 1

    return count


def create_H(points, w):

    H = np.sign(np.dot(np.transpose(w), np.transpose(points)))

    return H


def update_weights(w, sample, sampleIndex, Y, iteration):

    w = w + Y[sampleIndex] * sample

    iteration += 1

    return w, iteration


def create_points():

    points = np.random.uniform(-1, 1, (SAMPLESIZE, 2))

    # Add column of x0 = 1

    points = np.c_[np.ones(SAMPLESIZE), points]

    # NB! The points matrix has to be transposed,
    # it is not in the correct orientation for calculations

    return points


def create_X(points):

    X = np.transpose(points)

    return X


def learn(X, Y, w, H, RUNS):

    iterations = np.zeros(RUNS)

    for i in range(RUNS):

        select_random_point(points, Y, H)

    return iterations


def main():

    # Create random dataset of SAMPLESIZE samples

    points = create_points()

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
    H = create_H(points, w)

    iterations = learn(X, Y, w, H, RUNS)

if __name__ == "__main__":
    main()
