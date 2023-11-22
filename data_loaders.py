import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

FIG_SIZE = 5


def load_UNINet(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    lines = data.split('\n')

    N, NM = [int(value.split('=')[1]) for value in lines[1].split()]

    if "ROW LABELS:" in lines:
        labels_index = lines.index("ROW LABELS:")
        node_names = lines[labels_index + 1:labels_index + 1 + N]
    else:
        node_names = None
        
    # get level labels
    layer_labels_index = lines.index("LEVEL LABELS:")
    layer_labels = lines[layer_labels_index + 1:layer_labels_index + 1 + NM]
    
    start_index = lines.index("DATA:")
    MLG = []
    for i in range(NM):
        matrix_lines = lines[start_index + 1 + i * N:start_index + 1 + (i + 1) * N]
        matrix = np.array([list(map(int, line.split())) for line in matrix_lines])
        graph = nx.from_numpy_array(matrix)
        MLG.append(graph)

    return MLG, layer_labels, node_names


def display_MLG(MLG, layer_labels):

    M = len(MLG)
    
    fig, axes = plt.subplots(1, M, figsize=(FIG_SIZE * M, FIG_SIZE))

    for i, G in enumerate(MLG):
        nx.draw(G, with_labels=False, node_size=50, node_color='black', edge_color='black', ax=axes[i])
        axes[i].set_title(f"{layer_labels[i]}")

    plt.show()

    adjacency_matrices = [nx.adjacency_matrix(G) for G in MLG]
    fig, axes = plt.subplots(1, M, figsize=(FIG_SIZE * M, FIG_SIZE))
    for i, adjacency_matrix in enumerate(adjacency_matrices):
        axes[i].spy(adjacency_matrix, markersize=2, marker="D", color="Blue")
        axes[i].set_title(f"{layer_labels[i]}")
    plt.show()

# AUCS Dataset #####################################################
def init_graph():
    path = './datasets/AUCS/aucs_nodelist.txt'
    g = nx.Graph()
    with open(path) as f:
        for line in f:
            line = line.strip().split(',')
            if line[1] == 'NA':
                continue
            else:
                g.add_node(line[0])
    return g


def get_true_labels():
    true_labels = []
    na_list = []
    path = './datasets/AUCS/aucs_nodelist.txt'
    with open(path) as f:
        for line in f:
            line = line.strip().split(',')
            t = line[1]
            if t == 'NA':
                na_list.append(line[0])
            else:
                true_labels.append(int(t[-1])-1)
    return true_labels, na_list


def load_AUCS():
    path = './datasets/AUCS/aucs_edgelist.txt'

    # Declare each layer's graph
    lunch = init_graph()
    facebook = init_graph()
    leisure = init_graph()
    work = init_graph()
    coauthor = init_graph()
    table = {
        'lunch': lunch,
        'facebook': facebook,
        'leisure': leisure,
        'work': work,
        'coauthor': coauthor,
    }
    true_labels, na = get_true_labels()

    with open(path) as f:
        for line in f:
            line = line.strip().split(',')
            name = line[2]
            if line[0] in na or line[1] in na:
                continue
            else:
                table[name].add_edge(line[0], line[1])

    MLG = [lunch, facebook, leisure, work, coauthor]

    layer_labels = list(table.keys())

    return MLG, layer_labels, true_labels

