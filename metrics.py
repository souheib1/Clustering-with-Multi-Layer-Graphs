import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from itertools import combinations
from collections import Counter


def compute_purity(estimated_clusters, ground_truth_clusters, num_clusters, N):
    """
    Compute Purity to evaluate clustering performance
    """

    intersection_sizes = np.zeros((num_clusters, num_clusters))

    # Count intersections and fill the intersection_sizes matrix
    for i in range(num_clusters):
        for j in range(num_clusters):
            intersection_sizes[i, j] = np.count_nonzero(
                [obj in ground_truth_clusters[j] for obj in estimated_clusters[i]]
            )
    max_intersection_sizes = np.max(intersection_sizes, axis=1)
    purity = (1 / N) * np.sum(max_intersection_sizes)
    return purity


def compute_nmi(computed_clusters, ground_truth_clusters):
    """
    Compute Normalized Mutual Information (NMI) 
    
    """
    # Convert clusters to flat lists (to use scikit-learn's normalized_mutual_info_score)
    flat_computed = [item for sublist in computed_clusters for item in sublist]
    flat_ground_truth = [item for sublist in ground_truth_clusters for item in sublist]
    
    nmi = normalized_mutual_info_score(flat_ground_truth, flat_computed)
    return nmi


def compute_ri(computed_clusters, ground_truth_clusters):
    """
    Compute Rand Index (RI)
    """
    flat_computed = [item for sublist in computed_clusters for item in sublist]
    flat_ground_truth = [item for sublist in ground_truth_clusters for item in sublist]
    ri = adjusted_rand_score(flat_ground_truth, flat_computed)
    return ri


def purity_score(clustering_result, ground_truth):
    """
    Calculate purity score for a clustering result.

    Parameters:
    - clustering_result: Dictionary representing the clustering result, where keys are data points and values are cluster labels.
    - ground_truth: Dictionary representing the ground truth, where keys are data points and values are true class labels.

    Returns:
    - Purity score (float).
    """

    unique_clusters = np.unique(list(clustering_result.values()))

    N = len(ground_truth)

    # Initialize the total number of correctly assigned data points
    correctly_assigned = 0

    # Iterate over unique clusters
    for cluster_label in unique_clusters:
        # Get data points in the current cluster
        data_points_in_cluster = [data_point for data_point, label in clustering_result.items() if label == cluster_label]

        # Count the occurrences of true class labels in the current cluster
        class_counts = Counter([ground_truth[data_point] for data_point in data_points_in_cluster])

        # Get the most frequent true class label in the current cluster
        most_frequent_class = class_counts.most_common(1)[0][0]

        # Update the total correctly assigned data points
        correctly_assigned += class_counts[most_frequent_class]

    # Calculate purity score
    purity = correctly_assigned / N

    return purity

    
def nmi_score(clustering_result, ground_truth):
    """
    Calculate Normalized Mutual Information (NMI) score for a clustering result.

    Parameters:
    - clustering_result: Dictionary representing the clustering result, where keys are data points and values are cluster labels.
    - ground_truth: Dictionary representing the ground truth, where keys are data points and values are true class labels.

    Returns:
    - NMI score (float).
    """

    # Ensure the order of labels is consistent
    sorted_keys = sorted(clustering_result.keys())
    true_labels = [ground_truth[data_point] for data_point in sorted_keys]
    cluster_labels = [clustering_result[data_point] for data_point in sorted_keys]

    # Calculate NMI score
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)

    return nmi



def ri_score(clustering_result, ground_truth):
    """
    Calculate Rand Index (RI) for a clustering result.

    Parameters:
    - clustering_result: Dictionary representing the clustering result, where keys are data points and values are cluster labels.
    - ground_truth: Dictionary representing the ground truth, where keys are data points and values are true class labels.

    Returns:
    - Rand Index (RI) score (float).
    """
    
    # Get the number of data points
    num_data_points = len(ground_truth)

    # Initialize variables to count true positive, true negative, false positive, and false negative pairs
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    # Iterate over all pairs of data points
    for pair in combinations(range(num_data_points), 2):
        i, j = pair
        a = ground_truth[i] == ground_truth[j]  # True if the data points i and j have the same true class label
        b = clustering_result[i] == clustering_result[j]  # True if the data points i and j have the same cluster label

        if a and b:
            TP += 1
        elif a and not b:
            FN += 1
        elif not a and b:
            FP += 1
        elif not a and not b:
            TN += 1

    # Calculate Rand Index
    ri = (TP + TN) / (TP + TN + FP + FN)

    return ri
