import numpy as np
from collections import Counter


def clustering_accuracy(true_labels, clustering):
    categorical_to_numerical = {label: i for i, label in enumerate(np.unique(true_labels))}
    numerical_to_categorical = {i: label for label, i in categorical_to_numerical.items()}

    true_labels = [categorical_to_numerical[label] for label in true_labels]
    true_labels = np.array(true_labels)
    
    predicted_labels = np.array(list(clustering.values()))

    frequency_true = Counter(true_labels)
    frequency_predicted = Counter(predicted_labels)

    #  extract ordered keys by frequency
    frequency_true = [i[0] for i in frequency_true.most_common()]
    frequency_predicted = [i[0] for i in frequency_predicted.most_common()]

    map_labels = {i: j for i, j in zip(frequency_predicted,frequency_true)}
    mapped_labels = np.array([map_labels[i] for i in predicted_labels])

    # To match the labels with the clusters we want to maximize the diagonal of the confusion matrix
    accuracy = np.sum(true_labels == mapped_labels) / len(true_labels)

    new_clustering = {}
    for i, node in enumerate(clustering.keys()):
        new_clustering[node] = numerical_to_categorical[mapped_labels[i]]

    return accuracy, new_clustering
