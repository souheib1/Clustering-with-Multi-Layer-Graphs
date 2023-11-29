import numpy as np
import pandas as pd
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

# MIT Dataset #####################################################

    

def extract_friendship_graph():
    df = pd.read_csv('datasets/MIT/reality-mining-survey.txt', sep='\t')
    # start with 94 nodes from 0 to 89
    g = nx.Graph()
    g.add_nodes_from(range(90))
    g.add_weighted_edges_from(df[['id1', 'id2', 'close-friends?']].values)
    return g

def extract_proximity_graph():
    df = pd.read_csv('datasets/MIT/reality-mining-proximity.txt', sep='\t')
    df['start'] = pd.to_datetime(df['start'], format='mixed')
    df['end'] = pd.to_datetime(df['end'], format='mixed')

    df['duration'] = df['end'] - df['start']
    df['duration'] = df['duration'].dt.total_seconds()

    #  add a column to measure how many 30 mn intervals are in the duration of each proximity event
    df['duration_30mn'] = df['duration'] / 1800 + 1
    df['duration_30mn'] = df['duration_30mn'].astype(int)

    df = df[['id1', 'id2', 'duration_30mn']].groupby(['id1', 'id2']).sum().reset_index()
    df = df.rename(columns={'duration_30mn': 'proximity'})

    g = nx.Graph()
    g.add_nodes_from(range(90))

    g.add_weighted_edges_from(df[['id1', 'id2', 'proximity']].values)
    return g

def extract_calls_graph():
    df = pd.read_csv('datasets/MIT/reality-mining-calls.txt', sep='\t')
    df = df[df.subjectId != df.otherPartyId]
    incoming_mask = df['direction'] == 'Incoming'
    df.loc[incoming_mask, ['subjectId', 'otherPartyId']] = df.loc[incoming_mask, ['otherPartyId', 'subjectId']].values
    df.loc[incoming_mask, 'direction'] = 'Outgoing'
    df = df.drop_duplicates(subset=['subjectId', 'otherPartyId', 'direction', 'duration'])
    df = df.groupby(['subjectId', 'otherPartyId']).count().reset_index()
    df = df.rename(columns={'duration': 'calls'})

    df['id_pair'] = df.apply(lambda row: tuple(sorted([row['subjectId'], row['otherPartyId']])), axis=1)
    incoming_mask = df['direction'] == 'incoming'

    df.loc[incoming_mask, ['subjectId', 'otherPartyId']] = df.loc[incoming_mask, ['otherPartyId', 'subjectId']].values

    df.sort_values(by=['id_pair', 'direction'], inplace=True)
    df.drop_duplicates(subset=['id_pair', 'direction'], keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.groupby(['id_pair']).sum().reset_index()

    df['subjectId'] = df['id_pair'].apply(lambda x: x[0])
    df['otherPartyId'] = df['id_pair'].apply(lambda x: x[1])

    g = nx.Graph()
    g.add_nodes_from(range(90))
    g.add_weighted_edges_from(df[['subjectId', 'otherPartyId', 'calls']].values)
    return g

def extract_affiliation():
    affiliation = pd.read_csv('datasets/MIT/reality-mining-labels.txt', header=None)[0].tolist()
    return affiliation

def create_MIT():
    friendship = extract_friendship_graph()
    friendship_adj = pd.DataFrame(nx.adjacency_matrix(friendship).todense(), dtype=int)
    friendship_adj.to_csv("datasets/MIT/friendship.txt", header=None, index=None, sep=' ')

    calls = extract_calls_graph()
    calls_adj = pd.DataFrame(nx.adjacency_matrix(calls).todense(), dtype=int)
    calls_adj.to_csv("datasets/MIT/calls.txt", header=None, index=None, sep=' ')

    proximity = extract_proximity_graph()
    proximity_adj = pd.DataFrame(nx.adjacency_matrix(proximity).todense(), dtype=int)
    proximity_adj.to_csv("datasets/MIT/proximity.txt", header=None, index=None, sep=' ')

    affiliation = extract_affiliation()
    affiliation = pd.DataFrame(affiliation)
    affiliation.to_csv("datasets/MIT/affiliation.txt", header=None, index=None, sep=' ')


def load_MIT(process=False):
    if process:
        create_MIT()
    MLG = []
    friendship_adj = pd.read_csv('datasets/MIT/friendship.txt', header=None, sep=' ')
    friendship = nx.from_numpy_array(friendship_adj.values)
    MLG.append(friendship)

    calls_adj = pd.read_csv('datasets/MIT/calls.txt', header=None, sep=' ')
    calls = nx.from_numpy_array(calls_adj.values)
    MLG.append(calls)

    proximity_adj = pd.read_csv('datasets/MIT/proximity.txt', header=None, sep=' ')
    proximity = nx.from_numpy_array(proximity_adj.values)
    MLG.append(proximity)

    layer_labels = ['friendship', 'calls', 'proximity']

    true_labels = pd.read_csv('datasets/MIT/affiliation.txt', header=None, sep=' ')[0].tolist()

    return MLG, layer_labels, true_labels


