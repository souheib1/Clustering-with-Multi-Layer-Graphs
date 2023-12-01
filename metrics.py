import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


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
