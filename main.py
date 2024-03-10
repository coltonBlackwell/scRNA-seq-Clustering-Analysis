import anndata
import scanpy as sc
import numpy as np
from sklearn.decomposition import PCA as pca
import argparse
from kmeans import KMeans
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='number of clusters to find')
    parser.add_argument('--n-clusters', type=int,
                        help='number of features to use in a tree',
                        default=2)
    parser.add_argument('--data', type=str, default='data.csv',
                        help='data path')

    a = parser.parse_args()
    return(a.n_clusters, a.data)

def read_data(data_path):
    return anndata.read_csv(data_path)

def preprocess_data(adata: anndata.AnnData, scale :bool=True):
    """Preprocessing dataset: filtering genes/cells, normalization and scaling."""
    sc.pp.filter_cells(adata, min_counts=5000)
    sc.pp.filter_cells(adata, min_genes=500)

    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    adata.raw = adata

    sc.pp.log1p(adata)
    if scale:
        sc.pp.scale(adata, max_value=10, zero_center=True)
        adata.X[np.isnan(adata.X)] = 0

    return adata


def PCA(X, num_components: int):
    return pca(num_components).fit_transform(X)

def main():
    n_clusters, data_path = parse_args()
    heart = read_data(data_path)
    heart = preprocess_data(heart)
    X = PCA(heart.X, 100)


    kmeans = KMeans(n_clusters = n_clusters, init = 'random', max_iter = 300 ) #Change init = 'kmeans++' for KMeans++ implementation

    result = kmeans.fit(X)

    visualize_cluster(X, result, kmeans)
    # visualize_cluster_3d(X, result, kmeans)


def visualize_cluster(X: np.ndarray, clustering: np.ndarray, kmeans: np.ndarray):
    plt.figure(figsize=(8, 6))
    for cluster_index in range(kmeans.n_clusters):
        plt.scatter(X[clustering == cluster_index][:, 0], X[clustering == cluster_index][:, 1], 
                    label=f'Cluster {cluster_index}')
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], color='black', marker='o', s=50, label='Centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('KMeans Clustering - Best K (K = 10)')
    plt.legend()
    plt.show()


#Below function for 3D representation

def visualize_cluster_3d(X: np.ndarray, clustering: np.ndarray, kmeans: np.ndarray):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    for cluster_index in range(kmeans.n_clusters):
        ax.scatter(X[clustering == cluster_index][:, 0], X[clustering == cluster_index][:, 1], 
                X[clustering == cluster_index][:, 2], label=f'Cluster {cluster_index}')

    ax.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], kmeans.centroids[:, 2], color='black', marker='o', s=50, label='Centroids')

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title('Cluster Visualization')
    ax.legend()

    plt.show()


if __name__ == '__main__':
    main()
