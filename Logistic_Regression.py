import numpy as np

from .BaseClass import BaseClass


class LogisticRegression(BaseClass):
    
    """
        The init method is used to initialize the parameter
        Args:
            learning_rate     : to set the learning rate for gradient descent
            max_iters         : to set the maximum iterations for gradient descent 
         
        Returns:
            None              : None
    """
    def __init__(self, learning_rate, max_iters):
        self.lr = learning_rate
        self.max_iters = max_iters
        self.weights = None
        self.bias = None
        
        """
        The _get_prediction method is used to return the predictions after applying sigmoid
        Args:
            X          : an array of features of the test set 
            
        Returns:
            sigmoid    : signoid applied to the array of features
        """    

    def _get_prediction(self, X):
        linear = np.dot(X, self.weights) + self.bias
        sigmoid = 1 / (1 + np.exp(-linear))
        return sigmoid
    
        """
        The _initailize method is used to initilize the weights to zero
        Args:
            None         : None
            
        Returns:
            None    : None
        """ 
    
    def _initialize(self):
        self.weights = np.zeros(self.no_of_features)
        self.bias = 0
        
        """
        The _update_parameters method is used to update the weights and the bias.
        Args:
            d_w         : the derivate with respect to the weight
            d_b         : the derivative with respect to the bias
        Returns:
            None
        """  
    
    def _update_parameters(self, d_w, d_b):
        self.weights -= self.lr * d_w
        self.bias -= self.lr * d_b
        
        """
        The _get_gradients method is used to calculate the gradients w.r.t the weights and bias
        Args:
            X      : the array of features
            y      : the array of the variable of interest
            y_pred : the predictions
        Returns:
            d_w     : the derivate with respect to the weight
            d_b     : the derivative with respect to the bias
        """
        
    
    def _get_gradients(self, X, y, y_pred):
        error = y_pred - y
        d_w = (1 / self.no_of_samples) * np.dot(X.T, error)
        d_b = (1 / self.no_of_samples) * np.sum(error)
        return d_w, d_b
     
        """
        The fit method is used to fit the features X and y.
        Args:
            X      : the array of features
            y      : the array of the variable of interest
        Returns:
            None
        """
    
    
    def fit(self, X, y):
        self.no_of_samples, self.no_of_features = X.shape
        self._initialize()

        for _ in range(self.max_iters):

            y_pred = self._get_prediction(X)
            d_w, d_b = self._get_gradients(X, y, y_pred)
            self._update_parameters(d_w, d_b)
            
        """
        The predict method is used to predict the output for the features X.
        Args:
            X       : the array of features
        Returns:
            y_class : predcited output
        """    
    
    def predict(self, X):
        y_pred = self._get_prediction(X)
        y_class = [1 if p > 0.5 else 0 for p in y_pred]
        return y_class