import numpy as np
from .BaseClass import BaseClass

class KMeans:
    def __init__(self, k =3, tolerance = 0.0001, iters = 500):
        self.k = k
        self.tolerance = tolerance
        self.iters = iters

    def fit(self, data):
        """
        first we intialize the cluster_centroids, the first 'k' elements in the dataset will be our initial cluster_centroids.
        then we start the iterations to find the distance between the point and the cluster.
        choose the nearest centroid and average the cluster datapoints to re-calculate the cluster_centroids
        break out of the main loop if the cluster_centroids dont change indicating the optimal solution.
        
        Args:
            data  : training array of the feature variables
        
        """

        self.cluster_centroids = {}

        
        for i in range(self.k):
            self.cluster_centroids[i] = data[i]

        
        for i in range(self.iters):
            self.no_of_classes = {}
            for i in range(self.k):
                self.no_of_classes[i] = []

            
            for all_features in data:
                all_distances = [np.linalg.norm(all_features - self.cluster_centroids[centroid]) for centroid in            self.cluster_centroids]
                prediction = all_distances.index(min(all_distances))
                self.no_of_classes[prediction].append(all_features)

            previous = dict(self.cluster_centroids)

            
            for prediction in self.no_of_classes:
                self.cluster_centroids[prediction] = np.average(self.no_of_classes[prediction], axis = 0)

            Optimal_solution = True

            for centroid in self.cluster_centroids:

                original_centroid = previous[centroid]
                current = self.cluster_centroids[centroid]

                if np.sum((current - original_centroid)/original_centroid * 100.0) > self.tolerance:
                    Optimal_solution = False


            if Optimal_solution:
                break

    def pred(self, data):
        """
        first we intialize the cluster_centroids, the first 'k' elements in the dataset will be our initial cluster_centroids.
        then we start the iterations to find the distance between the point and the cluster.
        choose the nearest centroid and average the cluster datapoints to re-calculate the cluster_centroids
        break out of the main loop if the cluster_centroids dont change indicating the optimal solution.
        
        Args:
            data  : training array of the feature variables
            
        Returns:
            prediction : clustered data
        
        """
        all_distances = [np.linalg.norm(data - self.cluster_centroids[centroid]) for centroid in self.cluster_centroids]
        prediction = all_distances.index(min(all_distances))
        return prediction


