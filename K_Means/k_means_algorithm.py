import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_2D_clusters(clusters, features, title):
    """
    Plot a 2D visualization of clusters using the first two features.

    Parameters:
    ----------
    clusters : dict
        A dictionary where each key is a cluster ID and values are dictionaries containing 'centroid' and 'datapoints'.
    features : list of str
        The names of the features used for labeling the x and y axes.
    title : str
        The title for the plot.

    Returns:
    -------
    None
    """
    fig, ax = plt.subplots()

    colors = ['r', 'g', 'b', 'y', 'c', 'm']

    for cluster_id, cluster_data in clusters.items():
        points = np.array(cluster_data['datapoints'])
        if points.shape[0] > 0:
            ax.scatter(points[:, 0], points[:, 1], c=colors[cluster_id % len(colors)], label=f'Cluster {cluster_id}')

    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.legend()
    plt.title(title)
    plt.savefig('./Output/K_Means_Plot.png', format='png', dpi=300)  # Save the figure as a PNG file with high resolution
    plt.show()


def runKmeansAlgorithm(dataset, k, calculate_distance_function, rearrange_clusters_function, initiate_clusters_func):
    """
    Run the K-means algorithm on a dataset to cluster it into k clusters.

    Parameters:
    ----------
    dataset : pd.DataFrame
        The dataset on which K-means clustering is performed.
    k : int
        The number of clusters to create.
    calculate_distance_function : function
        A function to calculate the distance between two points (e.g., Euclidean distance).
    rearrange_clusters_function : function
        A function to update the centroids of clusters.
    initiate_clusters_func : function
        A function to initialize clusters with random centroids.

    Returns:
    -------
    dict
        A dictionary of clusters where each cluster contains its centroid and assigned datapoints.
    """
    clusters = initiate_clusters_func(k, dataset)
    old_clusters = clusters.copy()

    while True:
        # Assign datapoints to clusters
        clusters = loop_step_clusters_assignment(dataset, old_clusters, k, calculate_distance_function)
        # Update centroids
        new_clusters = rearrange_clusters_function(k, clusters, dataset)
        # Check for convergence
        if np.array_equal([clusters[c]['centroid'] for c in clusters],
                          [new_clusters[c]['centroid'] for c in new_clusters]):
            return old_clusters
        old_clusters = new_clusters.copy()


def initiate_clusters(k, dataset):
    """
    Initialize k clusters with random centroids.

    Parameters:
    ----------
    k : int
        Number of clusters to initialize.
    dataset : pd.DataFrame
        The dataset that provides the dimensionality for the centroids.

    Returns:
    -------
    dict
        A dictionary of clusters with centroids and an empty list for datapoints.
    """
    clusters = {}
    for centroid in range(k):
        center = 2 * (2 * np.random.random((dataset.shape[1],)) - 1)  # Random initialization of centroids
        cluster = {
            'centroid': center,
            'datapoints': []
        }
        clusters[centroid] = cluster
    return clusters


def calculate_distance_euclidean(point1, point2):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    ----------
    point1 : array-like
        First point.
    point2 : array-like
        Second point.

    Returns:
    -------
    float
        The Euclidean distance between the two points.
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))


def loop_step_clusters_assignment(dataset, clusters, k, calculate_distance_function):
    """
    Assign each datapoint in the dataset to the nearest cluster based on the minimum distance.

    Parameters:
    ----------
    dataset : pd.DataFrame
        The dataset with the datapoints to be assigned to clusters.
    clusters : dict
        Current clusters with their centroids.
    k : int
        Number of clusters.
    calculate_distance_function : function
        A function to calculate the distance between two points.

    Returns:
    -------
    dict
        Updated clusters with datapoints assigned to them based on proximity to centroids.
    """
    # Clear previous assignments
    for cluster in clusters.values():
        cluster['datapoints'] = []  # Reset the datapoints list for each cluster

    for idx in range(dataset.shape[0]):
        datapoint = dataset.iloc[idx].tolist()
        distances = [calculate_distance_function(datapoint, clusters[centroid]['centroid']) for centroid in range(k)]
        closest_cluster_id = np.argmin(distances)  # Assign to the closest cluster
        clusters[closest_cluster_id]['datapoints'].append(datapoint)

    return clusters


def rearrange_clusters(k, clusters, dataset):
    """
    Recalculate the centroids of each cluster by averaging the datapoints.

    Parameters:
    ----------
    k : int
        The number of clusters.
    clusters : dict
        The current clusters with their assigned datapoints.
    dataset : pd.DataFrame
        The dataset being clustered (used for dimensionality consistency).

    Returns:
    -------
    dict
        New clusters with updated centroids.
    """
    new_clusters = {}
    for i in range(k):
        points = np.array(clusters[i]['datapoints'])
        if points.shape[0] > 0:
            # Recalculate centroid as the mean of assigned points
            new_center = points.mean(axis=0)
        else:
            # If no points assigned to this cluster, reinitialize with a new random centroid
            new_center = 2 * (2 * np.random.random((dataset.shape[1],)) - 1)

        new_clusters[i] = {
            'centroid': new_center,
            'datapoints': []
        }
    return new_clusters


def kmeans():
    """
    Main function to load the dataset, run the K-means algorithm, and visualize the clusters.

    Returns:
    -------
    None
    """
    df = pd.read_csv('fruits.csv')
    features = ['Amount of Sugar', 'Price', 'Time it Lasts']
    sub_dataframe = df[features]

    # Run K-means algorithm
    clusters = runKmeansAlgorithm(sub_dataframe, 3, calculate_distance_euclidean, rearrange_clusters, initiate_clusters)

    # Plot the clusters using the first two features
    plot_2D_clusters(clusters, features[:2], 'Clusters Visualization - Clustered by Amount of Sugar and Price')


if __name__ == '__main__':
    kmeans() # run Kmeans

