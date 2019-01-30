import numpy as np


def combination(n, r):
    """
    Calculates nCr
    :param n:
    :param r:
    :return:
    """

    comb = factorial(n)/(factorial(r)*factorial(n-r))

    return int(comb)


def factorial(n):
    """
    Calculates n!
    :param n:
    :return:
    """

    fact = 1
    for i in range(1, n+1):
        fact *= i

    return fact


def q4_features(X, mode):
    # Given the data matrix X (where each row X[i,:] is an example), the function
    # computes the feature matrix B, where row B[i,:] represents the feature vector
    # associated to example X[i,:]. The features should be either linear or quadratic
    # functions of the inputs, depending on the value of the input argument 'mode'.
    # Please make sure to implement the features according to the *exact* order
    # specified in the text of the homework assignment.
    #
    # INPUT:
    #  X: a numpy.ndarray matrix of size [m x d] and type 'float' where each row
    #     is a d-dimensional input example
    #  mode: specifies the type of features;
    #        it is a 'str' that can be either 'linear' or 'quadratic'.
    #
    # OUTPUT:
    #  B: a numpy.ndarray matrix of size [m x n] and type 'float', with each row
    #     containing the feature vector of an example

    if mode == 'linear':

        # insert your code here
        B = np.full((X.shape[0], X.shape[1] + 1), 1.0)
        B[:, 1:] = X

    elif mode == 'quadratic':
        B = np.full((X.shape[0], X.shape[1] + 1), 1.0)
        B[:, 1:] = X

        xrow, xcol = X.shape
        temp = np.zeros([xrow, combination(xcol, 2) + xcol])

        for i in range(xrow):
            n = 0
            for j in range(xcol):
                f = X[i, j]
                for x in range(j, xcol):
                    temp[i, n] = f * X[i, x]
                    n += 1

        B = np.concatenate((B, temp), axis=1)

    else:
        print('Error, only linear and quadratic forms are supported')
        return []

    return B
