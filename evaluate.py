import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

import torch


def clustering_accuracy(true_labels, clustering, k):
    categorical_to_numerical = {label: i for i, label in enumerate(np.unique(true_labels))}
    numerical_to_categorical = {i: label for label, i in categorical_to_numerical.items()}

    true_labels = [categorical_to_numerical[label] for label in true_labels]

    true_labels = np.array(true_labels)
    predicted_labels = np.array(list(clustering.values()))

    # To match the labels with the clusters we want to maximize the diagonal of the confusion matrix
    cm = confusion_matrix(predicted_labels, true_labels)
    cm_argmax = cm.argmax(axis=0)
    new_clustering = np.array([cm_argmax[i] for i in true_labels])


    accuracy = np.sum(true_labels == new_clustering) / len(true_labels)


    new_clustering = {}
    for node, label in clustering.items():
        new_clustering[node] = numerical_to_categorical[clustering[label]]


    return accuracy, new_clustering
