import networkx as nx
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from evaluate import clustering_accuracy

from utils import compute_adjacency_matrix, compute_degree_matrix,compute_Laplacien

class SC_GED:
    """
    Spectral Clustering with Generalized Eigen-Decomposition (SC-GED).
        - MLG: List of networkx graph objects representing the multi-layer graph.
        - k: Target number of clusters.
        - most_informative : Index of the most informative layer.
        - alpha : parameter for stability .
        - beta: parameter for orthogonality .
        - P: matrix containing the set of joint eigenvectors as columns.
        - Q: enforced to be the inverse matrix of P.
    """
    
    def __init__(self, MLG, k=5, most_informative=0, alpha=0.5, beta=10): #changed most_informative by the list rank containing layer indexes in decreasing infomation 
        
        self.MLG = MLG
        self.most_informative = most_informative
        self.k = k
        self.n = MLG[0].number_of_nodes()
        self.M = len(MLG)

        self.alpha = alpha
        self.beta = beta

        self.W = torch.stack([compute_adjacency_matrix(G) for G in MLG])
        self.D = torch.stack([compute_degree_matrix(G) for G in MLG])
        self.L = compute_Laplacien(self.D, self.W, self.n, version="rw")

        self.Lambda, self.P, self.Q = self._eigen_decomposition()

        self.history = {
            "loss" : [],
            "data_fidelity" : [],
            "stability" : [],
            "orthogonality" : [],
        }

        self.clustering = None

    def objective_function(self, P, Q):
        P = P.reshape(self.n, self.n)
        Q = Q.reshape(self.n, self.n)
        S = self._data_fidelity(P, Q) + self._stability(P, Q) + self._orthogonality(P, Q)
        return S

    def _data_fidelity(self, P, Q):
        return 0.5 * torch.sum(torch.norm(self.L - torch.einsum('ij,kjl,lm->kim', P, self.Lambda, Q), dim=(-2, -1))**2)
    
    def _stability(self, P, Q):
        return 0.5 * self.alpha * (torch.norm(P) + torch.norm(Q))
    
    def _orthogonality(self, P, Q):
        return 0.5 * self.beta * torch.norm(P @ Q - torch.eye(self.n))

    def fit(self, n_iter=10_000):
        self._joint_eigen_decomposition(n_iter)

        self.U = self.P[:, :self.k]
        model = KMeans(n_clusters=self.k, random_state=0, n_init=10)
        model.fit(self.U)
        clustering = {}
        for i, node in enumerate(self.MLG[0].nodes()):
            clustering[node] = model.labels_[i]
        
        self.clustering = clustering

    def _eigen_decomposition(self):
        Lambda = []
        _, P = torch.linalg.eigh(self.L[self.most_informative])
        Q = torch.linalg.inv(P)

        for i in range(self.M):
            eigen_values = torch.linalg.eigvalsh(self.L[i])
            eigen_values.sort()
            Lambda.append(torch.diag(eigen_values))
        Lambda = torch.stack(Lambda, dim=0)
        return Lambda, P, Q

    def _joint_eigen_decomposition(self, n_iter):
        P = self.P.flatten().clone().detach().requires_grad_(True)
        Q = self.Q.flatten().clone().detach().requires_grad_(True)
        # LBFGS optimizer is used to have fast convergence as the matrices are sparse
        optimizer_P = optim.LBFGS([P], max_iter=1, line_search_fn="strong_wolfe")
        optimizer_Q = optim.LBFGS([Q], max_iter=1, line_search_fn="strong_wolfe")
        f_P = lambda P: self.objective_function(P, Q)
        f_Q = lambda Q: self.objective_function(P, Q)

        def closure_P():
            optimizer_P.zero_grad()
            loss = f_P(P)
            loss.backward()
            return loss
        
        def closure_Q():
            optimizer_Q.zero_grad()
            loss = f_Q(Q)
            loss.backward()
            return loss
                
        for i in tqdm(range(n_iter), desc="Joint Eigen Decomposition"):
            optimizer_P.step(closure_P)
            optimizer_Q.step(closure_Q)

            self.history["loss"].append(self.objective_function(P,Q).item())
            self.history["data_fidelity"].append(self._data_fidelity(P.reshape(self.n, self.n),Q.reshape(self.n, self.n)).item())
            self.history["stability"].append(self._stability(P.reshape(self.n, self.n),Q.reshape(self.n, self.n)).item())
            self.history["orthogonality"].append(self._orthogonality(P.reshape(self.n, self.n),Q.reshape(self.n, self.n)).item())
            
        self.P = P.detach().reshape(self.n, self.n)
        self.Q = Q.detach().reshape(self.n, self.n)

    def plot_loss(self):
        for key, value in self.history.items():
            plt.plot(value, label=key)

        plt.legend()
        plt.grid()
        plt.title("Loss history")
        plt.show()


    def evaluate(self, true_labels, verbose=False):
        assert self.clustering is not None, "You must fit the model before evaluating it."
        clustering = self.clustering
        best_accuracy, new_clustering = clustering_accuracy(true_labels, clustering, self.k)

        if verbose:
            print(f"norm of P: {torch.norm(self.P)}")
            print(f"norm of Q: {torch.norm(self.Q)}")
            print(f"norm of P @ Q - I: {torch.norm(self.P @ self.Q - torch.eye(self.n))}")
            print(f"Best accuracy: {best_accuracy}")

        return new_clustering, best_accuracy
