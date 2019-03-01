import numpy as np
from q4_train import q4_train
from q4_predict import q4_predict
from q4_mse import q4_mse


def q4_test_error(X, Y, Xtest, Ytest, lambdavec, mode):
    # Given training and test set, it trains the model and calculates the test error.
    #
    # INPUT
    #  X: a numpy.ndarray matrix of size [m x d] and type 'float' where each row
    #     is a d-dimensional input training example
    #  Y: a numpy.ndarray vector of size [m x 1] and type 'float', where the
    #     i-th element is the correct output value for the i-th input training example.
    #  Xtest: a numpy.ndarray vector of size [M x d] and type 'float', where
    #         each row is a d-dimensional test example
    #  Ytest: a numpy.ndarray vector of size [M x 1] and type 'float',
    #         containing the output values of the test examples
    #  lambdavec: a numpy.ndarray vector of size [k x 1] and type 'float'
    #             containing the set of regularization hyperparameter values
    #  mode: specifies the type of features;
    #        it is a 'str' that can be either 'linear' or 'quadratic'.
    #
    # OUTPUT
    #  error: a numpy.ndarray vector of size [k x 1] and type 'float'
    #         containing the test errors, one for each value in lambdavec.
    #

    # insert your code here

    errors = np.zeros(len(lambdavec))

    for i in range(len(lambdavec)):
        theta = q4_train(X, Y, [[lambdavec[i]]], mode)
        pred_Y = q4_predict(theta, X, mode)

        err = q4_mse(pred_Y, Y)
        errors[i] = err

    # expected size in checking_q4f.py (7) is different from the spec in this file [k x 1]

    return errors
