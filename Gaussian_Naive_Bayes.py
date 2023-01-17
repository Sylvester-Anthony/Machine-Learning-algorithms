import numpy as np
from .BaseClass import BaseClass


class GaussianNaiveBayes(BaseClass):
    def fit(self, X, y):
        """
         The fit method is used to fit the Gaussian Naive Bayes model to the training data
        Args:
            X          : an array of features
            Y          : an array of outcomes
         
        Returns:
            None              : None
        """
        self.number_of_samples, self.no_of_features = X.shape
        self.no_of_classes = len(np.unique(y))

        self.mean = np.zeros((self.no_of_classes, self.no_of_features))
        self.variance = np.zeros((self.no_of_classes, self.no_of_features))
        self.get_prior_probabilities = np.zeros(self.no_of_classes)

        for c in range(self.no_of_classes):
            X_c = X[y == c]

            self.mean[c, :] = np.mean(X_c, axis=0)
            self.variance[c, :] = np.var(X_c, axis=0)
            self.get_prior_probabilities[c] = X_c.shape[0] / self.number_of_samples

    def predict(self, X):
        """
        The predict method is used to return a set of predictions of the test set.
        Args:
            X          : an array of features
            
         
        Returns:
            y_hat      : a numpy array of predictions
        
        """
        y_hat = [self.get_class_probability(x) for x in X]
        return np.array(y_hat)

    def get_class_probability(self, x):
        """
        to get class probabilities of a particular class
        
        Args:
        X          : an array of features
            
        Returns:
        posterior_probabilities : the posterior probability
          
        """
        posterior_probabilities = list()

        for c in range(self.no_of_classes):
            mean = self.mean[c]
            variance = self.variance[c]
            prior_probabilities = np.log(self.get_prior_probabilities[c])

            posterior_probability = np.sum(np.log(self.gaussian_density(x, mean, variance)))
            posterior_probability = prior_probabilities + posterior_probability
            posterior_probabilities.append(posterior_probability)

        return np.argmax(posterior_probabilities)

    def gaussian_density(self, x, mean, var):
        """
         Definition of the gaussian density function
        """
        constant = 1 / np.sqrt(var * 2 * np.pi)
        prob_gaussian = np.exp(-0.5 * ((x - mean) ** 2 / var))

        return constant * prob_gaussian