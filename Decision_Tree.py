import numpy as np
from .BaseClass import BaseClass


class Node:
    """
     Class definition to build the node of a decision tree
    """
    def __init__(self, feature=None, threshold=None, left_split=None, right_split=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left_split = left_split
        self.right_split = right_split
        self.value = value
    
    def is_leaf(self):
        return self.value is not None



class DecisionTree(BaseClass):
    """
    Decision Tree class
    """
     
    def __init__(self, maximum_depth=100, minimum_samples_split=2):
        """
        function to iniatialize the tree 
        """
        self.maximum_depth = maximum_depth
        self.minimum_samples_split = minimum_samples_split
        self.root = None

    def is_completed(self, depth):
        
        """
        function to check if the building of the tree is completed
        Args:
            depth = depth of the tree
         
        Returns:
           bool
        
        
        """
        if (depth >= self.maximum_depth
            or self.no_of_class_labels == 1
            or self.no_of_samples< self.minimum_samples_split):
            return True
        return False
    
    def entropy(self, y):
        
        """
        function to calculate the entropy 
        
        Args:
            y          : the feature of interest
        Returns:
           entropy     : the entropy
        
        """
        
        get_proportion = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in get_proportion if p > 0])
        return entropy
    
    
    def information_gain(self, X, y, threshold):
        """
        function to calculate the information gain
        
        Args:
            X          : an array of features
            threshold  : the decision threshold value
            y          : the feature of interest
           

        """
        get_parent_loss = self.entropy(y)
        left_split_index, right_split_index = self.create_split(X, threshold)
        n, number_left_split, number_right_split = len(y), len(left_split_index), len(right_split_index)

        if number_left_split == 0 or number_right_split == 0: 
            return 0
        
        get_child_loss = (number_left_split / n) * self.entropy(y[left_split_index]) + (number_right_split / n) * self.entropy(y[right_split_index])
        return get_parent_loss - get_child_loss
    
    
    def best_split(self, X, y, features):
        """
        function to perform best split based on the information gain
        Args:
        X          : an array of features
        y          : the feature of interest
        Returns:
        split feature and split threshold
        
        """
        split = {'score':- 1, 'feat': None, 'threshold': None}

        for feat in features:
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for threshold in thresholds:
                score = self.information_gain(X_feat, y, threshold)

                if score > split['score']:
                    split['score'] = score
                    split['feat'] = feat
                    split['threshold'] = threshold

        return split['feat'], split['threshold']
    
    
    

    def create_split(self, X, threshold):
        
        """
        function to perform a split
        Args:
            X          : an array of features
            threshold  : the decision threshold value
         
        Returns:
           left_split_index, right_split_index
        """
        left_split_index = np.argwhere(X <= threshold).flatten()
        right_split_index = np.argwhere(X > threshold).flatten()
        return left_split_index, right_split_index


    
    def build_tree(self, X, y, depth=0):
        """
        function to build a tree
        
        Args:
            X          : an array of features
            Y          : an array of outcomes
         
        Returns:
            None              : None
        
        """
        self.no_of_samples, self.no_of_features = X.shape
        self.no_of_class_labels = len(np.unique(y))

       
        if self.is_completed(depth):
            most_common_label = np.argmax(np.bincount(y))
            return Node(value=most_common_label)

       
        random_features = np.random.choice(self.no_of_features, self.no_of_features, replace=False)
        best_feature, best_threshold = self.best_split(X, y, random_features)

     
        left_split_index, right_split_index = self.create_split(X[:, best_feature], best_threshold)
        left_split_child = self.build_tree(X[left_split_index, :], y[left_split_index], depth + 1)
        right_split_child = self.build_tree(X[right_split_index, :], y[right_split_index], depth + 1)
        return Node(best_feature, best_threshold, left_split_child, right_split_child)
    
    def traverse_tree(self, x, node):
        """
        function to traverse a tree
        
        Args:
        x        : an array of features
        node     : node of a tree
         
        Returns:
         A traversed tree
        
        """
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left_split)
        return self.traverse_tree(x, node.right_split)

    def fit(self, X, y):

        """
        A method to fit the data and build the tree
        
        Args:
        X          : an array of features
        y         : an array of outcomes
         
        Returns:
        None              : None
        """
        self.root = self.build_tree(X, y)

    def predict(self, X):
        """
        A method to preditc the test data based on the fitted tree
        
        Args:
        X          : an array of features
                  
        Returns:
        predictions      : a numpy array of predictions
        
        """
        predictions = [self.traverse_tree(x, self.root) for x in X]
        return np.array(predictions)
