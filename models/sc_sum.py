import sys
sys.path.append("../scripts")

from sklearn.cluster import KMeans
import torch

from metrics import purity_score, nmi_score, ri_score
from utils import compute_adjacency_matrix, compute_degree_matrix


class SC_SUM:
    """
    Spectral Clustering applied on the summation of the adjacency matrices.
        - MLG: List of networkx graph objects representing the multi-layer graph.
        - k: Target number of clusters.
        - most_informative : Index of the most informative layer.
        - P: matrix containing the set of joint eigenvectors as columns.
    """
    
    def __init__(self, MLG, k=5): 
        
        self.MLG = MLG
        self.k = k
        self.n = MLG[0].number_of_nodes()
        self.M = len(MLG)

        self.W = torch.stack([compute_adjacency_matrix(G) for G in MLG])
        self.D = torch.stack([compute_degree_matrix(G) for G in MLG])

        self.P = self._eigen_decomposition()

        self.clustering = None



    def fit(self):
        self.U = self.P[:, :self.k]
        model = KMeans(n_clusters=self.k, random_state=0, n_init=100)
        model.fit(self.U)
        clustering = {}
        for i, node in enumerate(self.MLG[0].nodes()):
            clustering[node] = model.labels_[i]
        
        self.clustering = clustering

    def _eigen_decomposition(self):
        self.D_sqrt = torch.stack([torch.diag(1/torch.sqrt(Di)) for Di in self.D])
        self.sum_W = torch.sum(self.D_sqrt@self.W@self.D_sqrt, dim=0)
        self.L = torch.eye(self.n) - self.sum_W

        _, P = torch.linalg.eigh(self.L)
        return P

    def evaluate(self, true_labels, verbose=False):
        assert self.clustering is not None, "You must fit the model before evaluating it."
        if isinstance(list(self.clustering.keys())[0], int):
            nodes = sorted(self.clustering.keys())
        else:
            nodes = sorted(self.clustering.keys(), key=lambda x: int(x[1:]))
        
        clustering = {}
        for i, node in enumerate(nodes):
            clustering[i] = self.clustering[node]


        N = len(true_labels)
        ground_truth_clustering = {i: true_labels[i] for i in range(N)}

        purity = purity_score(clustering, ground_truth_clustering)
        nmi = nmi_score(clustering, ground_truth_clustering)
        ri = ri_score(clustering, ground_truth_clustering)



        if verbose:
            print("Purity: ", purity)
            print("NMI: ", nmi)
            print("RI: ", ri)


        return purity, nmi, ri