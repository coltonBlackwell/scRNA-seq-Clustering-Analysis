import numpy as np

class KMeans():

    def __init__(self, n_clusters: int, init: str='random', max_iter = 300):
        """

        :param n_clusters: number of clusters
        :param init: centroid initialization method. Should be either 'random' or 'kmeans++'
        :param max_iter: maximum number of iterations
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None # Initialized in initialize_centroids()

    def fit(self, X: np.ndarray):

        np.random.seed(42)

        self.initialize_centroids(X)

        iteration = 0
        clustering = np.zeros(X.shape[0])

        while iteration < self.max_iter:

            clustering = self.assign_clusters(X)
            new_centroids = self.update_centroids(clustering, X)
            
            if np.allclose(self.centroids, new_centroids):
                break 

            self.centroids = new_centroids
            iteration += 1

        # print("NEW centroids", self.centroids)
        # print("clustering: ", clustering)
        print("Result: ", self.silhouette(clustering, X))

        return clustering

    def allclose(a, b, rtol=1e-05, atol=1e-08):

        return np.all(np.isclose(a, b, rtol=rtol, atol=atol))

    def update_centroids(self, clustering: np.ndarray, X: np.ndarray):
        new_centroids = np.zeros_like(self.centroids)
        for i in range(self.n_clusters):
            new_centroids[i] = np.mean(X[np.nonzero(clustering == i)[0]], axis=0)
        return new_centroids
    

    def initialize_centroids(self, X: np.ndarray):
        """
        Initialize centroids either randomly or using kmeans++ method of initialization.
        :param X:
        :return:
        """
        if self.init == 'random':
            self.centroids = self.kmeans_centroids(X)
        elif self.init == 'kmeans++':
            self.centroids = self.kmeansplusplus_centroids(X)
        else:
            raise ValueError('Centroid initialization method should either be "random" or "k-means++"')

    def kmeans_centroids(self, X: np.ndarray):
        n_samples, _ = X.shape
        centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        return centroids

    def kmeansplusplus_centroids(self, X: np.ndarray):
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))
        centroids[0] = X[np.random.randint(n_samples)]

        for i in range(1, self.n_clusters):
            distances = self.euclidean_distance(X, centroids[:i])

            min_distances_to_centroids = np.min(distances, axis=1)

            probabilities = min_distances_to_centroids**2 / np.sum(min_distances_to_centroids**2)
            next_centroid_index = np.random.choice(np.arange(n_samples), p=probabilities)
            centroids[i] = X[next_centroid_index]

        return centroids
    

    def assign_clusters(self, X: np.ndarray):

        distances = self.euclidean_distance(X, self.centroids)
        return np.argmin(distances, axis=1)

    def euclidean_distance(self, X1:np.ndarray, X2:np.ndarray):
        """
        Computes the euclidean distance between all pairs (x,y) where x is a row in X1 and y is a row in X2.
        Tip: Using vectorized operations can hugely improve the efficiency here.
        :param X1:
        :param X2:
        :return: Returns a matrix `dist` where `dist_ij` is the distance between row i in X1 and row j in X2.
        """
        dist_sq = np.sum((X1[:, np.newaxis, :] - X2) ** 2, axis=2)
    
        dist = np.sqrt(dist_sq)
        
        return dist

    def silhouette(self, clustering: np.ndarray, X: np.ndarray):
        n_samples = X.shape[0]
        v = 0 

        distances = self.euclidean_distance(X, X)

        for i in range(n_samples):
            cluster_label = clustering[i]

            intra_cluster_distance = np.mean(distances[i, clustering == cluster_label])
            other_cluster_distances = np.mean(distances[i, clustering != cluster_label])
            
            if intra_cluster_distance < other_cluster_distances:
                v += (other_cluster_distances - intra_cluster_distance) / max(intra_cluster_distance, other_cluster_distances)

        v /= n_samples

        return v