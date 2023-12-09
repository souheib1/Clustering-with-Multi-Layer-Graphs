import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from itertools import combinations
from collections import Counter
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def purity_score(clustering_result, ground_truth):
    """
    Calculate purity score for a clustering result.
    """
    unique_clusters = np.unique(list(clustering_result.values()))
    N = len(ground_truth)
    correctly_assigned = 0
    # Iterate over unique clusters
    for cluster_label in unique_clusters:
        data_points_in_cluster = [data_point for data_point, label in clustering_result.items() if label == cluster_label]
        class_counts = Counter([ground_truth[data_point] for data_point in data_points_in_cluster])
        most_frequent_class = class_counts.most_common(1)[0][0]
        correctly_assigned += class_counts[most_frequent_class]
    purity = correctly_assigned / N
    return purity

    
def nmi_score(clustering_result, ground_truth):
    """
    Calculate Normalized Mutual Information (NMI) score for a clustering result.
    """
    # Ensure the order of labels is consistent
    sorted_keys = sorted(clustering_result.keys())
    true_labels = [ground_truth[data_point] for data_point in sorted_keys]
    cluster_labels = [clustering_result[data_point] for data_point in sorted_keys]
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    return nmi


def ri_score(clustering_result, ground_truth):
    """
    Calculate Rand Index (RI) for a clustering result.
    """
    num_data_points = len(ground_truth)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    # Iterate over all pairs of data points
    for pair in combinations(range(num_data_points), 2):
        i, j = pair
        a = ground_truth[i] == ground_truth[j] 
        b = clustering_result[i] == clustering_result[j]  
    
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


def compute_confusion_matrix(ground_truth, predicted_clusters, cmap=plt.cm.Blues):
    """
    Compute the confusion matrix of a predicted clustering
    """  
    true_labels = list(ground_truth.values())
    pred_labels = list(predicted_clusters.values())
    unique_labels = sorted(set(true_labels) | set(pred_labels))
    ground_truth_labels = [true_labels.index(label) for label in true_labels]
    predicted_labels = [pred_labels.index(label) for label in pred_labels]
    conf_matrix = confusion_matrix(ground_truth_labels, predicted_labels)

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap=cmap, xticklabels=unique_labels, yticklabels=unique_labels, cbar=False)
    plt.xlabel('Predicted Labels', fontsize=14)
    plt.ylabel('True Labels', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16)
    plt.tick_params(axis='both', which='both', length=0) 
    plt.tight_layout()
    plt.show()
