import numpy as np

### Functions for you to fill in ###

def closed_form(X, Y, lambda_factor):
    """
    Computes the closed form solution of linear regression with L2 regularization

    Args:
        X - (n, d + 1) NumPy array (n datapoints each with d features plus the bias feature in the first dimension)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        lambda_factor - the regularization constant (scalar)
    Returns:
        theta - (d + 1, ) NumPy array containing the weights of linear regression. Note that theta[0]
        represents the y-axis intercept of the model and therefore X[0] = 1
    """
    # YOUR CODE HERE
    # from problem definition
    # theta = ((X^T*X + lambda*I)^-1)  *  X^T*Y

    d = X.shape[1]
    I = np.identity(d)
    Xt = np.transpose(X)
    lf_I = lambda_factor * I
    XtX = np.matmul(Xt,X)
    XtY = np.matmul(Xt,Y)
    A = XtX + lf_I
    A_inv = np.linalg.inv(A)

    return(np.matmul(A_inv,XtY))
    # raise NotImplementedError

### Functions which are already complete, for you to use ###

def compute_test_error_linear(test_x, Y, theta):
    test_y_predict = np.round(np.dot(test_x, theta))
    test_y_predict[test_y_predict < 0] = 0
    test_y_predict[test_y_predict > 9] = 9
    return 1 - np.mean(test_y_predict == Y)
