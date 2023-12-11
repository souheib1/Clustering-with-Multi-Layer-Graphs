import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch


def degree_distribution(G,aff=False):
    #Compute and optionally visualize the degree distribution of a graph.
    histogram = nx.degree_histogram(G) 
    if aff:
        # Degree distribution
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        axes[0].plot(histogram,'r',marker='o')
        axes[0].set_ylabel("Frequency")
        axes[0].set_xlabel("Degree")
        axes[0].set_title("Degree Distribution (Linear Scale)")
        # loglog plot
        axes[1].loglog(histogram,'b',marker='o')
        axes[1].set_ylabel("Log Frequency")
        axes[1].set_xlabel("Log Degree")
        axes[1].set_title("Degree Distribution (Log-Log Scale)")
    return(histogram)


def spectral_clustering(G, k):
    # Perform spectral clustering to partition graph G into k clusters
    A = nx.adjacency_matrix(G)
    n = G.number_of_nodes()
    degrees = [G.degree(node) for node in G.nodes()]
    D = np.diag(degrees)
    D_inv = np.linalg.pinv(D)
    L_rw = np.eye(n) - D_inv @ A
    eigen_values, eigen_vectors = eigs(L_rw, k=k, which='SR')
    eigen_vectors = eigen_vectors.real
    model = KMeans(n_clusters=k)
    model.fit(eigen_vectors)
    # clustering ={}
    # for i,node in enumerate (G.nodes()) :
    #     clustering[node] = model.labels_[i] 
    return model.labels_



def modularity(G, clustering):
    #Compute the modularity of a graph based on a given  clustering
    m = G.number_of_edges()
    cluster_ids = set(clustering.values()) 
    modularity = 0

    for id in cluster_ids:
        cluster_nodes = [node for node in G.nodes() if clustering[node] == id]
        subgraph_id = G.subgraph(cluster_nodes)
        lc = subgraph_id.number_of_edges() 
        degrees = [G.degree(node) for node in cluster_nodes] 
        dc = np.sum(degrees)
        modularity = modularity + (lc/m - ((dc*dc)/(4*m*m))) 
    return modularity

def compute_adjacency_matrix(G):
    W = nx.adjacency_matrix(G)
    W = torch.tensor(W.toarray(), dtype=torch.float)
    return W

def compute_degree_matrix(G):
    D = torch.tensor([G.degree(node) for node in G.nodes()])
    D = torch.add(D, 1e-4)
    return D

def compute_Laplacien(D, W, n, version="rw"):
    if version=="rw":
        M = len(D)
        Dinv = torch.stack([torch.diag(1/diag) for diag in D])
        I_Mn = torch.eye(n).repeat(M, 1, 1)
        L = I_Mn - torch.einsum('ijk,ikl->ijl', Dinv, W)
    elif version=="sym":
        if len(W)!=len(D):
            print("Sanity Check failed, check the dimenstion of the input")
            version = "rw"
            print("Random Walk Laplacien is returned instead")
            return compute_Laplacien(D, W, n, version=version)
        else:
            M = len(D)
            I_Mn = torch.eye(n).repeat(M, 1, 1)
            D_sqrt = torch.stack([torch.diag(1/torch.sqrt(diag)) for diag in D])
            L = I_Mn - torch.einsum('ijk,ikl,ilm->ijm', D_sqrt, W, D_sqrt)
    else:
        raise NotImplementedError("version not supported")
    return L


def compute_adjacency_matrix(G):
    W = nx.adjacency_matrix(G)
    W = torch.tensor(W.toarray(), dtype=torch.float)
    return W

def compute_degree_matrix_from_adjMatrix(W):
    """
    This function enables to extract the degree matrix from the weighted 
    adjacency matrix W 
    """
    n = W.shape[0]
    D = torch.zeros(n, n, dtype=torch.float32)

    for i in range(n):
        d = torch.sum(W[i] > 0)
        D[i, i] = d

    return D


def layer_ranks(G):
    """
    Returns a list of layer indexes in decreasing infomation order 
    """
    l=[]
    #to be implemented
    return l

def get_metric_values(metric, results, dict_layer_dataset=None): 
    values = np.zeros(6)
    values[::2] = results.loc[:, ('SC_GED', metric)]
    for i, (dataset_name, model_name) in enumerate(dict_layer_dataset.items()):
        values[2*i+1] = results.loc[dataset_name, (model_name, metric)]
    values = np.round(values, 2)
    return values

def dispaly_results(results, dict_layer_dataset=None):
    if dict_layer_dataset is None:
        dict_layer_dataset = {
        'Cora':"Layer1",
        'AUCS':"Layer0",
        'MIT':"Layer2"
        } 
    metrics = ['purity', 'nmi', 'ri']
    for metric in metrics:
        values = get_metric_values(metric, results, dict_layer_dataset)
        plt.rcParams["figure.figsize"] = (8,5)

        datasets = ("Cora", "AUCS", "MIT")
        purity_results = {
            "Single-Layer Eigen-Decomposition ": values[1::2],
            "Generalized Eigen-Decomposition": values[::2],
        } 


        x = np.arange(len(datasets))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained')

        for attribute, measurement in purity_results.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)

            ax.bar_label(rects, padding=3)
            multiplier += 1
        
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(f'{metric.capitalize()} Score', fontsize=16)
        ax.set_xticks(x + width, datasets)
        # put legend on the right with 1 row for each attribute outside the plot
        ax.legend(loc='upper left', ncols=1, fontsize=16, bbox_to_anchor=(0, 1.3))
        ax.set_ylim(0, 1)

        plt.show()


def evaluation_pipeline(dataset, model):
    model.fit()
    purity, nmi, ri = model.evaluate(dataset.labels, verbose=False)
    return purity, nmi, ri