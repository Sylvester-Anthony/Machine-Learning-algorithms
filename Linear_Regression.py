import numpy as np
import random
import math
from .BaseClass import BaseClass


class LinearRegression(BaseClass):

    def fit(self, X, y, method, learning_rate , iters, batch_size):
        """
        The fit method is used to fit the training data and get the best parameters after the fit.
        We implement both the anlytical solution as well as stochastic gradient descent 
        Args:
            X                 : training array of the feature variables
            y                 : training array of the variable of interest
            method            : analytical or Stochastic Gradient Descent 
            learning rate     : to set the learning rate for SGD
            iters        : maximum iters
            batch_size.       : batch size for SGD
        Returns:
            weights           : weights matrix after the fit 
        """
        X = np.concatenate([X, np.ones_like(y)], axis=1)
        no_of_rows, no_of_columns = X.shape
        
        if method == 'analytical':
            if no_of_rows >= no_of_columns == np.linalg.matrix_rank(X):
                self.weights = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)),X.transpose()),y)
            else:
                print('X has not full column rank')

        elif method == 'sgd':
            self.weights = np.random.normal(scale=1 / no_of_columns, size=(no_of_columns, 1))
            for i in range(iters):
                X_y = np.concatenate([X, y], axis=1)
                np.random.shuffle(X_y)
                X, y = X_y[:, :-1], X_y[:, -1:]
                for j in range(int(np.ceil(no_of_rows / batch_size))):
                    start, end = batch_size * j, np.min([batch_size * (j + 1), no_of_rows])
                    X_batch, y_batch = X[start:end], y[start:end]
                    gradient = 2 * np.matmul(X_batch.transpose(),
                        (np.matmul(X_batch,self.weights)- y_batch))
                    self.weights -= learning_rate * gradient
        else:
            print(f'Unknown method: \'{method}\'')

        return self

    def predict(self, X):
        """
        Args:
            X             : testing array of the feature variables
        Returns:
            predictions   : X multiplied with the weights matrix
        """
        if not hasattr(self, 'weights'):
            print('Cannot predict')
            return

        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

        if X.shape[1] != self.weights.shape[0]:
            print('Shapes do not match')
            return

        return np.matmul(X, self.weights)