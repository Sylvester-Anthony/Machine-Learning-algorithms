import numpy as np
from .BaseClass import BaseClass

class SVM(BaseClass):

    def __init__(self, learning_rate, lambda_parameter, iters):
        """
        method to initialize the algorithm
    
        Args:
        
        learning_rate     :  to set the learning rate for the gradient descent 
        lambda_paramter   :  the regularization parameter
        iters             :  the number of iterations 
      
        """
        self.lr = learning_rate
        self.lambda_parameter = lambda_parameter
        self.iters = iters
        self.weights = None
        self.bias = None

    def init_weights_bias(self, X):
        """
        method to initialize the weights and bias
    
        Args:
        
        X    :  a matrix of features
        
        """
        no_of_features = X.shape[1]
        self.weights = np.zeros(no_of_features)
        self.bias = 0

    def map_classes(self, y):
        """
        method to map the classes based on the variable of interest
    
        Args:
        
        y    :  a matrix of features of the variable of interest
        
        """
        return np.where(y <= 0, -1, 1)

    def check_satisfy_get_cont(self, x, idx):
        """
        method to check if the constrainsts are satisfied
        
    
        Args:
        
        x    :  a matrix of features
        idx  : id of x


        """
        linear_model = np.dot(x, self.weights) + self.bias 
        return self.class_map[idx] * linear_model >= 1
    
    def compute_gradients(self, get_con, x, idx):
        """
        method to initialize the weights and bias
    
        Args:
        
        X    :  a matrix of features
        idx  : id of x
        
        Returns:
        d_weights : the derivative of weights 
        d_bias    : the derivative of bias


        """
        if get_con:
            d_weights = self.lambda_parameter * self.weights
            d__bias = 0
            return d_weights, d__bias
        
        d_weights = self.lambda_parameter * self.weights - np.dot(self.class_map[idx], x)
        d__bias = - self.class_map[idx]
        return d_weights, d__bias
    
    def update_weights_bias(self, d_weights, d__bias):
        """
        method to update the bias and the weights
        
        Args :
        d_weights : the derivative of weights 
        d_bias    : the derivative of bias
        
        """
        self.weights -= self.lr * d_weights
        self.bias -= self.lr * d__bias
    
    def fit(self, X, y):
        """
        method to fit the model to the data
        
        Args:
        
        X    :  a matrix of features
        y    :  a matrix of features of the variable of interest
        
        """
        self.init_weights_bias(X)
        self.class_map = self.map_classes(y)

        for _ in range(self.iters):
            for idx, x in enumerate(X):
                get_con = self.check_satisfy_get_cont(x, idx)
                d_weights, d__bias = self.compute_gradients(get_con, x, idx)
                self.update_weights_bias(d_weights, d__bias)
    
    def predict(self, X):
        """
        method to predict the data
        
        Args:
        
        X    :  a matrix of features
        
        Returns:
        prediction   :  predicted value 
        
        """
        unsigned_pred = np.dot(X, self.weights) + self.bias
        prediction = np.sign(unsigned_pred)
        return np.where(prediction == -1, 0, 1)