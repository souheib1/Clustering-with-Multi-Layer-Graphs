import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt



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
    D_inv = diags([1/d for d in degrees])
    L_rw = np.eye(n) - D_inv @ A
    eigen_values, eigen_vectors = eigs(L_rw, k=k, which='SR')
    eigen_vectors = eigen_vectors.real
    model = KMeans(n_clusters=k)
    model.fit(eigen_vectors)
    clustering ={}
    for i,node in enumerate (G.nodes()) :
        clustering[node] = model.labels_[i] 
    return clustering



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