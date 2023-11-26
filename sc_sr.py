import networkx as nx
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from utils import compute_adjacency_matrix,compute_degree_matrix_from_adjMatrix

def SC_SR(W, k, _lambda,ranking):
    """
    clustering on multi-layer graphs with the Clustering with 
    spectral regularization method.
    
    Arguments:
        - W : weighted adjacency tensor of size M x n x n.
        - k : Target number of cluster.
        - _lambda : list of regularization parameters of M-1 layers.
        - ranking : list of indexes of layers sorted from most informative to the least one
    """
    
    #Sanity check and transform to adjancency matrix if needed 
    if isinstance(W, nx.Graph): 
        W = compute_adjacency_matrix(W)
        
    # 1) Input
    M, n = W.size(0), W.size(1)
    informative_layer = ranking[0]
    W_1 = W[informative_layer, :, :]

    #  2) For G(1), compute the degree matrix D(1)
    D_1 = compute_degree_matrix_from_adjMatrix(W_1)

    # 3) Compute the random walk graph Laplacian Lrw(1)
    Lrw_1 = torch.pinverse(D_1) @ (D_1 - W_1)

    # 4) Compute the first k eigenvectors (u1..uk) of L(1)
    w, v = torch.eig(Lrw_1, eigenvectors=True)
    wk_arg = torch.argsort(w[:, 0])[:k]

    # 5) Let U in Rn×k be the matrix containing (u1..uk) ascolumns
    U = v[:, wk_arg]

    # 6) For i = 2.. n, solve the spectral regularization problem in Eq. (11) 
    # for each ui and replace it with the solution fi in U to 
    # form the new low dimensional embedding U''
    
    for p in ranking[1:]:
        W_p = W[p, :, :]
        D_p = compute_degree_matrix_from_adjMatrix(W_p)
        L_sym = torch.pinverse(torch.sqrt(D_p)) @ (D_p - W_p) @ torch.pinverse(torch.sqrt(D_p)) #(D^(1/2) @ L @ D^(1/2)) 
        lambda_p = _lambda[p - 1] # no lambda for most informative so shift by 1 index
        mu = 1 / lambda_p
        for i in range(k):
            U[:, i] = mu * torch.pinverse(L_sym + mu * torch.eye(n)) @ U[:, i] # closed form solution f_i 
  

    # 7) Cluster yi in Rk into C1,..,Ck using the K-means algorithm
    kmeans = KMeans(n_clusters=k, random_state=0).fit(U.detach().numpy())
    return kmeans.labels_

