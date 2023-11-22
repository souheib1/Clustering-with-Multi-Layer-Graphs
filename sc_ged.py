import networkx as nx
from sklearn.cluster import KMeans

from tqdm import tqdm

import torch
import torch.optim as optim


def compute_degree_matrix(G):
    # Compute the degree matrix of a graph
    D = [G.degree(node) for node in G.nodes()]
    return torch.add(torch.tensor(D), 1e-4)

def compute_Laplacien(D, W, n, version="rw"):
    if version=="rw":
        Dinv = torch.diag(1 / D)
        L = torch.eye(n) - Dinv @ W
    elif version=="sym":
        Dinv = torch.diag(1 / torch.sqrt(torch.tensor(D, dtype=torch.float)))
        L = Dinv @ (torch.diag(torch.tensor(D, dtype=torch.float)) - W) @ Dinv
    else:
        raise NotImplementedError("version not supported")
    return L

class SC_GED:
    def __init__(self, MLG, k=5, most_informative=0, alpha=0.5, beta=10):
        self.MLG = MLG
        self.most_informative = most_informative

        self.k = k
        self.n = MLG[0].number_of_nodes()
        self.M = len(MLG)

        self.alpha = alpha
        self.beta = beta

        self.W = [torch.tensor(nx.adjacency_matrix(G).toarray(), dtype=torch.float) for G in MLG]
        self.D = [compute_degree_matrix(G) for G in MLG]
        self.L = torch.stack([compute_Laplacien(self.D[i], self.W[i], self.n, version="rw") for i in range(self.M)], dim=0)
        self.Lamb, self.P, self.Q = self._eigen_decomposition()

        self.loss_history = []

    def objective_function(self, P, Q):
        P = P.reshape(self.n, self.n)
        Q = Q.reshape(self.n, self.n)
        data_fidelity = 0.5 * torch.sum(torch.norm(self.L - torch.einsum('ij,kjl,lm->kim', P, self.Lamb, Q), dim=(-2, -1))**2)
        sparsity = 0.5 * self.alpha * (torch.norm(P) + torch.norm(Q))
        orthogonality = 0.5 * self.beta * torch.norm(P @ Q - torch.eye(self.n))
        S = data_fidelity + sparsity + orthogonality
        return S


    def __call__(self, n_iter=10_000):
        self._joint_eigen_decomposition(n_iter)

        self.U = self.P[:, :self.k]
        model = KMeans(n_clusters=self.k, random_state=0, n_init=10)
        model.fit(self.U)
        clustering = {}
        for i, node in enumerate(self.MLG[0].nodes()):
            clustering[node] = model.labels_[i]
        return clustering

    def _eigen_decomposition(self):
        Lamb = []
        print("start decomposition")
        _, P = torch.linalg.eigh(self.L[self.most_informative])
        Q = torch.linalg.inv(P)

        for i in tqdm(range(self.M), desc="Eigen Decomposition"):
            eigen_values = torch.linalg.eigvalsh(self.L[i])
            eigen_values.sort()
            Lamb.append(torch.diag(eigen_values))
        Lamb = torch.stack(Lamb, dim=0)
        return Lamb, P, Q

    def _joint_eigen_decomposition(self, n_iter):
        P = self.P.flatten().clone().detach().requires_grad_(True)
        Q = self.Q.flatten().clone().detach().requires_grad_(True)
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

            self.loss_history.append(self.objective_function(P,Q).item())
            
        self.P = P.detach().reshape(self.n, self.n)
        self.Q = Q.detach().reshape(self.n, self.n)