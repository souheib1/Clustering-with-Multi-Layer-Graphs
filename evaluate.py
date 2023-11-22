import numpy as np
import matplotlib.pyplot as plt
import itertools

import torch


def accuracy_with_label_mapping(mapping, true_labels, clustering):
    true_labels = np.array(true_labels)
    predicted_labels = np.array(list(clustering.values()))
    predicted_labels_mapped = np.array([mapping[value] for value in predicted_labels])
        
    return np.sum(true_labels == predicted_labels_mapped) / len(true_labels)


def clustering_accuracy(true_labels, clustering, k):
    labels_range = list(range(k))
    all_possible_mappings = [dict(zip(perm, labels_range)) for perm in itertools.permutations(labels_range, k)]

    best_accuracy = 0
    best_mapping = None

    for mapping in all_possible_mappings:
        acc = accuracy_with_label_mapping(mapping, true_labels, clustering)
        if acc > best_accuracy:
            best_accuracy = acc
            best_mapping = mapping


    new_clustering = {}
    for node, label in clustering.items():
        new_clustering[node] = best_mapping[label]

    return best_accuracy, new_clustering


def evaluate_model(model, true_labels, n_iter=1_000):
    clustering = model(n_iter)

    print(f"norm of P: {torch.norm(model.P)}")
    print(f"norm of Q: {torch.norm(model.Q)}")
    print(f"norm of P @ Q - I: {torch.norm(model.P @ model.Q - torch.eye(model.n))}")

    best_accuracy, new_clustering = clustering_accuracy(true_labels, clustering, model.k)
    print(f"Best accuracy: {best_accuracy}")

    loss_history = {
        "objective_function": model.loss_history,
        "data_fidelity": model.data_fidelity_history,
        "stability": model.stability_history,
        "orthogonality": model.orthogonality_history,
    }

    return new_clustering, loss_history, best_accuracy

def plot_loss(loss_history):
    for key, value in loss_history.items():
        plt.plot(value, label=key)

    plt.legend()
    plt.grid()
    plt.title("Loss history")
    plt.show()
