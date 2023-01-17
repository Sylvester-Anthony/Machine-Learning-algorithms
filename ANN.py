import numpy as np
from .BaseClass import BaseClass


EPS = np.finfo(np.float64).eps

"""

Convert the variables to one-hot encoded variables

"""

def to_categorical(classes):
    n_classes = classes.max()+1
    y = np.zeros((classes.shape[0], n_classes))
    
    for i in range(classes.shape[0]):
        y[i, classes[i]] = 1
    
    return y

"""
 Define the activation functions
 
"""


def identity(x):
    return x

def d_identity(x):
    return np.tile(np.identity(x.shape[1]), (x.shape[0], 1, 1))

def relu(x):
    return np.maximum(x, 0)

def d_relu(x):
    return np.vectorize(lambda v: 1 if v > 0 else 0)(x)

def softmax(x):
    x = x - x.max(axis=1).reshape((-1, 1))
    exp = np.exp(x)
    s = np.sum(exp, axis=1).reshape((-1, 1))
    return exp/(s+EPS)

def d_softmax(x):
    s = softmax(x)
    D = np.stack([np.diag(s[i, :]) for i in range(s.shape[0])], axis=0)
    comb = np.matmul(np.expand_dims(s, 2), np.expand_dims(s, 1))
    return D - comb

"""
Define the loss function and their derivatives

"""
def mean_squared_error(y_pred, y_true):
    return np.mean((y_pred-y_true)**2, axis=1).reshape((-1, 1))

def d_mean_squared_error(y_pred, y_true):
    return np.expand_dims((2/y_pred.shape[1])*(y_pred-y_true), 1)

def categorical_crossentropy(y_pred, y_true):
    return -np.log(np.sum(y_true*y_pred, axis=1)+EPS)

def d_categorical_crossentropy(y_pred, y_true):
    return np.expand_dims(-y_true/(y_pred+EPS), 1)

""" 
Define SGD with momentum
"""
class SGD:
    def __init__(self, learning_rate, momentum):
        self.learning_rate = learning_rate
        self.momentum = momentum
    
    def update(self, old_parameters, gradient):
        if not hasattr(self, 'delta_parameters'):
            self.delta_parameters = np.zeros_like(old_parameters)
        
        self.delta_parameters = self.momentum*self.delta_parameters - self.learning_rate*gradient
        new_parameters = old_parameters + self.delta_parameters
        
        return new_parameters


class NeuralNetwork(BaseClass):
    

    
    def __init__(self, layers, hidden_activation, output_activation, loss, optimizer):
        """
        Parameters:
            layers: a list consisting of the number of nodes in each layer (including input and output layers)
                    e.g.: [5, 20, 5] denotes 5 inputs in the input layer, 20 nodes in hidden layer, and 25 output nodes
                    
            hidden_activation: activation of hidden layers(activation_function, its_derivative)
                    e.g.: (relu, d_relu)
                    
            output_activation: activation of output layer; (activation_function, its_derivative)
          
                    
            loss: a tuple of form (loss_function, its_derivative)
                    e.g.: (categorical_crossentropy, d_categorical_crossentropy)
                    
            optimizer: an object with a method .update(old_parameters, gradient) which returns the new parameters
                    e.g.: SGD()
        """
        self.layers = layers
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss = loss
        self.optimizer = optimizer
        
        self.weights = []
        self.biases = []
        self.nlayers = len(layers)
        nrows = layers[0]
        for i in range(1, self.nlayers):
            ncols = layers[i]
            std_dev = np.sqrt(1/(nrows+ncols)) 
            self.weights.append(np.random.normal(size=(nrows, ncols), scale=std_dev))
            self.biases.append(np.random.normal(size=(1, ncols), scale=std_dev))
            nrows = ncols
    
    def flatten_parameters(self, weights, biases):
        """
        method to flatten the parameters 
        
        Args:
        
        weights : weights matrix
        
        biases  : bias matrix
        
        """
        params = []
        for W in weights:
            params.append(W.flatten())
        for b in biases:
            params.append(b.flatten())
        
        params = np.concatenate(params)
        
        return params
    
    def restore_parameters(self, params):
        
        """
        method to restore the parameters 
        
        Args:
        
        params : a matrix of the parameter calculated
        
        """
        weights = []
        biases = []
        
        start = 0
        for i in range(1, self.nlayers):
            nrows = self.layers[i-1]
            ncols = self.layers[i]
            end = start+nrows*ncols
            p = params[start:end]
            W = p.reshape((nrows, ncols))
            weights.append(W)
            start = end
        
        for i in range(1, self.nlayers):
            ncols = self.layers[i]
            end = start+ncols
            p = params[start:end]
            b = p.reshape((1, ncols))
            biases.append(b)
            start = end
        
        return (weights, biases)
    
    def forward_propagation(self, x):
        """
        method to calculate forward propagation
        
        Args:
        
        x   : a matrix of the data features
        
        """
        io_arrays = []
        for i in range(self.nlayers):
            if i > 0:
                x = np.matmul(x, self.weights[i-1]) + self.biases[i-1]
            layer_io = [x] 
            if i == self.nlayers-1:
                activation = self.output_activation[0]
            elif i > 0:
                activation = self.hidden_activation[0]
            else:
                activation = lambda v: v
            x = activation(x)
            layer_io.append(x) 
            io_arrays.append(layer_io)
        return io_arrays
    
    def backward_propagation(self, io_arrays, y_true):
        """
        method to calculate backward propagation
        
        Args:
        
        weights : matrix of values from forward propagation
        
        y_true  : matrix of y values
        
        """
        
        
        e = self.loss[1](io_arrays[-1][1], y_true)
        
        batch_size = y_true.shape[0]
        d_weights = []
        d_biases = []
        for i in range(self.nlayers-1, 0, -1):
            if i == self.nlayers-1:
                e = np.matmul(e, self.output_activation[1](io_arrays[i][0]))
                e = np.squeeze(e, 1)
            else:
                e = e * self.hidden_activation[1](io_arrays[i][0])
            d_w = np.matmul(io_arrays[i-1][1].transpose(), e)/batch_size
            d_b = np.mean(e, axis=0)
            d_weights.append(d_w)
            d_biases.append(d_b)
            e = np.matmul(e, self.weights[i-1].transpose())
        
        d_weights.reverse()
        d_biases.reverse()
        
        return (d_weights, d_biases)
    
    def fit(self, x, y, batch_size, epochs, categorical=False):
        
        """
        method to fit the network to the model
        
        Args:
        
        x            : matrix of features
        
        y            : matrix of variable of interest
        
        batch_size   : batch size 
        
        epochs       : number of epochs
        
        categorical  : to denote if y is  categorical data
        
        """
        
        
        if categorical:
            y = to_categorical(y)
        
        y_ncols = y.shape[1]
        
        n_samples = x.shape[0]
        
        epoch_loss = []
        for i in range(epochs):
            xy = np.concatenate([x, y], axis=1)
            np.random.shuffle(xy)
            x, y = np.split(xy, [-y_ncols], axis=1)
            
            print(f'Epoch {i+1}/{epochs}\n')
            start = 0
            loss_hist = []
            while start < n_samples:
                end = min(start+batch_size, n_samples)
                x_batch = x[start:end, :]
                y_batch = y[start:end, :]
                
                io_arrays = self.forward_propagation(x_batch)
                d_weights, d_biases = self.backward_propagation(io_arrays, y_batch)
                
                params = self.flatten_parameters(self.weights, self.biases)
                gradient = np.nan_to_num(self.flatten_parameters(d_weights, d_biases))
                
                params = self.optimizer.update(params, gradient)
                
                self.weights, self.biases = self.restore_parameters(params)
                
                loss_hist.append(np.mean(self.loss[0](io_arrays[-1][1], y_batch)))
                print(f'{end}/{n_samples} ; loss={np.mean(loss_hist)}', end='\r')
                if end >= n_samples:
                    print('\n')
                start = end
            epoch_loss.append(np.mean(loss_hist))
        return np.array(epoch_loss)
    
    def predict(self, x, classes=False):
        
        """
        method to predict the data
        
        Args:
        
        x       : matrix of features
        
        classes  : False if the task is regression and true if the class is classification
        
        Returns:
        
        output : predicition
        
        """
        
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
            
        output = self.forward_propagation(x)[-1][1]
        
        if classes:
            return np.argmax(output, 1)
        else:
            return output

    
   







